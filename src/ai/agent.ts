/**
 * Core AI Agent Runner
 *
 * Each agent is a tool-loop agent: generateText with maxSteps allows the model
 * to call tools iteratively until it finishes or hits the step limit.
 * Every ~8 seconds, each agent gets a fresh tick with updated dynamic context.
 */

import { generateText, type CoreMessage } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

import type { AgentConfig, RegionConfig } from "@/engine/types";
import { useAgentsStore } from "@/store/agents";
import { useChatsStore } from "@/store/chats";

import { scheduleAgentCall } from "./rate-limiter";
import { getTimeContext, getCurrentGameTime } from "./context/time";
import { getNearbyContext } from "./context/nearby";

import { createMoveTools } from "./tools/move";
import { createChatTools } from "./tools/chat";
import { createDoorTools, getDoorContext } from "./tools/door";
import { createMemoryTool, MemoryStore } from "./tools/memory";
import {
  createRelationshipTools,
  RelationshipState,
} from "./tools/relationship";
import { getPointsContext } from "./tools/points";

import { getPrisonerPrompt } from "@/scenarios/prison/prompts/prisoner";
import { getGuardPrompt } from "@/scenarios/prison/prompts/guard";

// --- Persistent message log entry (never trimmed) ---

interface MessageLogEntry {
  agentId: string;
  agentName: string;
  agentRole: string;
  currentRegion: string;
  role: string;
  content: string;
  timestamp: number;
}

/** Append-only log of all LLM messages across the simulation. Never trimmed. */
const messageLog: MessageLogEntry[] = [];

// --- Hourly C-score snapshots ---

interface CScoreSnapshot {
  simulationTime: string;
  realTimestamp: number;
  scores: Array<{ id: string; name: string; points: number; region: string }>;
}

const cScoreSnapshots: CScoreSnapshot[] = [];
let lastSnapshotHour: number | null = null;

/**
 * Check whether the simulation clock has crossed an hour boundary since
 * the last snapshot, and if so, record a C-score snapshot.
 */
function checkHourlyCScoreSnapshot(): void {
  const simTime = getCurrentGameTime();
  if (!simTime) return;

  const currentHour = simTime.getHours();
  if (lastSnapshotHour === null) {
    // First call — record the starting hour but don't snapshot yet
    lastSnapshotHour = currentHour;
    return;
  }

  if (currentHour === lastSnapshotHour) return;

  // Hour changed — take a snapshot
  lastSnapshotHour = currentHour;

  const agentsStore = useAgentsStore.getState();
  const prisoners = agentsStore
    .getAllAgents()
    .filter((a) => a.role === "prisoner");

  const hours = simTime.getHours();
  const minutes = simTime.getMinutes().toString().padStart(2, "0");
  const ampm = hours >= 12 ? "PM" : "AM";
  const h12 = hours % 12 || 12;

  cScoreSnapshots.push({
    simulationTime: `${h12}:${minutes} ${ampm}`,
    realTimestamp: Date.now(),
    scores: prisoners.map((p) => ({
      id: p.id,
      name: p.name,
      points: p.points,
      region: getAgentRegion(p.id),
    })),
  });

  console.log(
    `[AI] C-Score snapshot at ${h12}:${minutes} ${ampm}:`,
    prisoners.map((p) => `${p.name}=${p.points}`).join(", "),
  );
}

// --- State per agent ---

interface AgentRuntime {
  config: AgentConfig;
  systemPrompt: string;
  messages: CoreMessage[];
  memoryStore: MemoryStore;
  relationshipState: RelationshipState;
  running: boolean;
}

const agentRuntimes = new Map<string, AgentRuntime>();

/**
 * Track how many consecutive ticks each agent has spent in the same chat.
 * After MAX_CHAT_TICKS, the agent is auto-removed from the chat to prevent
 * the infinite chat loop where all agents gather and stop moving.
 */
const chatTickCounts = new Map<string, { chatId: string; ticks: number }>();
const MAX_CHAT_TICKS = 6;

/**
 * The initial user message every agent starts with. Also used as the
 * reset anchor if runtime.messages gets irrecoverably corrupted.
 */
const INITIAL_USER_MESSAGE =
  "The simulation has started. Look around, decide what to do, and take action using the tools available to you. You MUST use at least one tool (like move_to_region) on every turn.";

/**
 * Trim a message history without breaking tool-call / tool-result pairing.
 *
 * The OpenAI-compatible API requires every `tool` role message to be
 * preceded by an `assistant` message containing the matching tool_call_id.
 * A naive `slice(-N)` can cut between an assistant with tool_calls and
 * its tool results, leaving an orphan `tool` message at index 0 — which
 * causes every subsequent request to 400 permanently for that agent.
 *
 * This helper walks forward from the proposed cut point until it finds a
 * non-`tool` message, guaranteeing the kept window starts at a valid
 * boundary (user, system, or assistant).
 */
function safeTrimMessages(
  messages: CoreMessage[],
  keepLast: number,
): CoreMessage[] {
  if (messages.length <= keepLast) return messages;
  let cutIndex = messages.length - keepLast;
  while (cutIndex < messages.length && messages[cutIndex].role === "tool") {
    cutIndex++;
  }
  return messages.slice(cutIndex);
}

/**
 * Detect whether a message history is corrupted in a way that will cause
 * every subsequent API call to fail — specifically, starting with an
 * orphan `tool` message.
 */
function isMessageHistoryCorrupted(messages: CoreMessage[]): boolean {
  return messages.length > 0 && messages[0].role === "tool";
}

/**
 * Reasoning-model text-to-tool-call fallback.
 *
 * Qwen3.6 and other thinking models occasionally emit text like
 * `"say: Hello there."` instead of calling the `say` tool. If the agent
 * is in an active chat and no `say` tool was fired this tick, parse the
 * message out of the text and send it through the chat store directly.
 *
 * Returns the recovered message string on success, or null if nothing
 * was recovered.
 */
function recoverSayFromText(params: {
  agentId: string;
  agentName: string;
  text: string | undefined;
  toolCallsThisTick: Array<{ toolName: string }>;
}): string | null {
  const { agentId, agentName, text, toolCallsThisTick } = params;
  if (!text) return null;

  // If the model already called `say` this tick, don't double-send.
  if (toolCallsThisTick.some((tc) => tc.toolName === "say")) return null;

  const agent = useAgentsStore.getState().getAgent(agentId);
  if (!agent?.currentChatId) return null;

  // Match `say:` (case-insensitive), optionally wrapped in quotes, at the
  // start of the text or on its own line. Allow a leading newline from
  // the reasoning model's output conventions.
  const match = text.match(/(?:^|\n)\s*say\s*[:\-]\s*["']?(.+?)["']?\s*$/is);
  if (!match) return null;

  const message = match[1].trim();
  if (!message) return null;

  const chatsStore = useChatsStore.getState();
  const sendResult = chatsStore.sendMessage(agent.currentChatId, {
    id: agentId,
    name: agentName,
    content: message,
    timestamp: Date.now(),
  });
  if (!sendResult.success) return null;

  // Notify chat partners to tick sooner, same as the real `say` tool.
  notifyChatPartners(agent.currentChatId, agentId);

  return message;
}

// --- Bridge functions (set by the Phaser engine) ---

export interface BridgeFunctions {
  moveTo: (agentId: string, x: number, y: number) => Promise<boolean>;
  forceMoveTo: (
    guardId: string,
    prisonerId: string,
    x: number,
    y: number,
  ) => Promise<boolean>;
  findDoorByRegions: (
    r1: string,
    r2: string,
  ) => {
    door: unknown;
    lock: (d: unknown) => boolean;
    unlock: (d: unknown) => boolean;
  } | null;
  getAllDoorStates: () => Array<{
    region1: string;
    region2: string;
    isLocked: boolean;
  }>;
  getRegions: () => RegionConfig[];
  getAgentWorldPosition: (agentId: string) => { x: number; y: number } | null;
}

let bridgeFns: BridgeFunctions | null = null;

export function setBridgeFunctions(fns: BridgeFunctions) {
  bridgeFns = fns;
  console.log(
    "[AI] Bridge functions set. Regions available:",
    fns.getRegions().length,
  );
}

// --- LLM model (any OpenAI-compatible endpoint) ---

const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL_ID = "openrouter/free";

/**
 * Hard ceiling on a single tick's LLM request.
 *
 * The endpoint is serverless: after it scales to zero, the first request
 * cold-starts a worker (~90s) before responding, so this must sit
 * comfortably above that. Its real job is to stop a black-holed request
 * from permanently silencing an agent — each agent's rate limiter allows
 * only one in-flight call (maxConcurrent: 1), so a request that never
 * resolves would freeze that agent for the rest of the session. On
 * timeout the call throws (aborting the fetch); the catch block in
 * tickAgent then reschedules after a short backoff, and the retry lands
 * on the now-warm worker.
 */
const REQUEST_TIMEOUT_MS = 150_000;

const modelCache = new Map<
  string,
  ReturnType<ReturnType<typeof createOpenAI>>
>();

function getModel(role: string) {
  const modelId =
    role === "guard"
      ? import.meta.env.VITE_GUARD_MODEL || DEFAULT_MODEL_ID
      : import.meta.env.VITE_PRISONER_MODEL || DEFAULT_MODEL_ID;

  const cached = modelCache.get(modelId);
  if (cached) return cached;

  const apiKey = import.meta.env.VITE_OPENROUTER_API_KEY || "";
  const baseURL = import.meta.env.VITE_OPENAI_BASE_URL || DEFAULT_BASE_URL;
  if (!apiKey) {
    console.error(
      "[AI] VITE_OPENROUTER_API_KEY is not set! Agents will not work.",
    );
  }
  const provider = createOpenAI({
    baseURL,
    apiKey,
  });
  const model = provider(modelId);
  modelCache.set(modelId, model);
  return model;
}

// --- System prompt builder ---

function buildSystemPrompt(agentConfig: AgentConfig): string {
  const number = agentConfig.name.replace(/[^0-9]/g, "") || "1";

  if (agentConfig.role === "guard") {
    const prisoners = useAgentsStore
      .getState()
      .getAllAgents()
      .filter((a) => a.role === "prisoner")
      .map((p) => p.name.replace(/[^0-9]/g, ""))
      .filter(Boolean)
      .join(", ");
    return getGuardPrompt(number, prisoners || "1, 2, 3, 4, 5, 6");
  }

  return getPrisonerPrompt(number);
}

function buildDynamicContext(agentId: string, runtime: AgentRuntime): string {
  const sections: string[] = [];

  sections.push(getTimeContext());
  sections.push(getNearbyContext(agentId));
  sections.push(runtime.memoryStore.getContext());
  sections.push(runtime.relationshipState.getContext());

  // Points context
  const agentsStore = useAgentsStore.getState();
  sections.push(
    getPointsContext({
      agentId,
      role: runtime.config.role,
      getPoints: (id) => useAgentsStore.getState().getPoints(id),
      getAllPrisonerPoints: () =>
        useAgentsStore.getState().getAllPrisonerPoints(),
    }),
  );

  // Door states
  if (bridgeFns) {
    sections.push(
      getDoorContext({ getAllDoorStates: bridgeFns.getAllDoorStates }),
    );
  }

  // Available regions (so the agent knows what move targets exist)
  // Filter out "Escape" — agents shouldn't navigate there directly
  if (bridgeFns) {
    const regions = bridgeFns.getRegions().filter((r) => r.label !== "Escape");
    if (regions.length > 0) {
      const regionNames = regions.map((r) => r.label).join(", ");
      sections.push(`[Available Regions] ${regionNames}`);
    }
  }

  // --- Chat timeout: force-leave if agent has been chatting too long ---
  const agent = agentsStore.getAgent(agentId);
  if (agent?.currentChatId) {
    const tracker = chatTickCounts.get(agentId);
    if (tracker && tracker.chatId === agent.currentChatId) {
      tracker.ticks++;
    } else {
      chatTickCounts.set(agentId, { chatId: agent.currentChatId, ticks: 1 });
    }
    const current = chatTickCounts.get(agentId)!;
    if (current.ticks >= MAX_CHAT_TICKS) {
      console.log(
        `[AI] ${agentId}: Auto-leaving chat ${agent.currentChatId} after ${current.ticks} ticks`,
      );
      useChatsStore.getState().leaveSession(agent.currentChatId, agentId);
      chatTickCounts.delete(agentId);
      // After force-leaving, fall through to the "not in chat" branch below
    }
  } else {
    // Not in a chat — reset tracker
    chatTickCounts.delete(agentId);
  }

  // Re-read agent state after possible force-leave
  const agentAfterTimeout = agentsStore.getAgent(agentId);

  // Chat context — this is the critical section for back-and-forth conversation
  if (agentAfterTimeout?.currentChatId) {
    const chatsStore = useChatsStore.getState();
    const session = chatsStore.getAgentSession(agentId);
    if (session) {
      const participantNames = session.participants
        .filter((pid) => pid !== agentId)
        .map((pid) => agentsStore.getAgent(pid)?.name ?? pid)
        .join(", ");

      const messages = session.messages;
      if (messages.length > 0) {
        const chatLines = messages
          .slice(-10)
          .map((m) => `${m.name}: ${m.content}`);
        const lastMsg = messages[messages.length - 1];
        const lastSpeakerIsMe = lastMsg.id === agentId;

        sections.push(
          `[ACTIVE CONVERSATION with ${participantNames}]\n` +
            `${chatLines.join("\n")}\n` +
            (lastSpeakerIsMe
              ? `(You spoke last. Wait for a response, or use leave_chat if done.)`
              : `(${lastMsg.name} just spoke. You MUST respond using the "say" tool now.)`),
        );
      } else {
        sections.push(
          `[ACTIVE CONVERSATION with ${participantNames}]\n` +
            `(Conversation just started. Use the "say" tool to greet them.)`,
        );
      }
    }
  } else {
    // Not in a chat — check if someone nearby might want to talk
    const chatsStore = useChatsStore.getState();
    const nearby = chatsStore.getNearbyAgents(agentId);
    const nearbyInChat = nearby.filter((n) => n.inChat);
    if (nearbyInChat.length > 0) {
      const names = nearbyInChat.map((n) => n.name).join(", ");
      sections.push(
        `[Note] ${names} ${nearbyInChat.length === 1 ? "is" : "are"} in a conversation nearby. You could use start_chat to join.`,
      );
    }
  }

  return sections.filter(Boolean).join("\n\n");
}

// --- Tool composition ---

/**
 * Build tools for an agent. All deps use fresh getState() calls so they
 * always read the latest store values (not stale snapshots).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildTools(
  agentId: string,
  runtime: AgentRuntime,
): Record<string, any> {
  if (!bridgeFns) {
    console.warn(`[AI] Bridge not ready, no tools for ${agentId}`);
    return {};
  }

  const bf = bridgeFns;

  const baseTools = {
    ...createMoveTools({
      agentId,
      getRegions: () => bf.getRegions(),
      moveTo: bf.moveTo,
      forceMoveTo: runtime.config.role === "guard" ? bf.forceMoveTo : undefined,
      onMoveStart: (id, label, isForced, targetId) => {
        useAgentsStore.getState().updateMoveBubble(id, {
          content: `${isForced ? "🔗" : "🚶"} ${label}`,
          timestamp: Date.now(),
          duration: 5000,
          isForced,
        });
        if (isForced && targetId) {
          useAgentsStore.getState().updateMoveBubble(targetId, {
            content: `🔗 ${label}`,
            timestamp: Date.now(),
            duration: 5000,
            isForced: true,
          });
        }
      },
    }),
    ...createChatTools({
      agentId,
      agentName: runtime.config.name,
      getCurrentChatId: () =>
        useAgentsStore.getState().getAgent(agentId)?.currentChatId ?? null,
      getNearbyAgents: () => useChatsStore.getState().getNearbyAgents(agentId),
      createChat: (ids) => useChatsStore.getState().createSession(ids),
      joinChat: (chatId) =>
        useChatsStore.getState().joinSession(chatId, agentId),
      leaveChat: (chatId) =>
        useChatsStore.getState().leaveSession(chatId, agentId),
      sendMessage: (chatId, msg) =>
        useChatsStore.getState().sendMessage(chatId, msg),
      getMessages: (chatId) => useChatsStore.getState().getMessages(chatId),
      onMessageSent: notifyChatPartners,
      canAdjustCScore: runtime.config.role === "guard",
      getChatParticipants: (chatId) => {
        const session = useChatsStore
          .getState()
          .getAllSessions()
          .find((s) => s.id === chatId);
        if (!session) return [];
        const store = useAgentsStore.getState();
        return session.participants
          .map((pid) => store.getAgent(pid))
          .filter((a): a is NonNullable<typeof a> => !!a)
          .map((a) => ({ id: a.id, name: a.name, role: a.role }));
      },
      adjustCScore: (prisonerId, delta) => {
        const store = useAgentsStore.getState();
        if (delta >= 0) store.addPoints(prisonerId, delta);
        else store.subtractPoints(prisonerId, -delta);
        return store.getPoints(prisonerId);
      },
    }),
    ...createMemoryTool(runtime.memoryStore),
    ...createRelationshipTools(runtime.relationshipState),
  };

  // Guard-only tools
  if (runtime.config.role === "guard") {
    Object.assign(
      baseTools,
      createDoorTools({
        agentId,
        findDoorByRegions: bf.findDoorByRegions,
        getAllDoorStates: bf.getAllDoorStates,
        moveTo: bf.moveTo,
      }),
    );
  }

  return baseTools;
}

// --- Tick scheduling ---

/** Track pending fast-tick timers so we can avoid duplicates. */
const pendingFastTicks = new Set<string>();

/**
 * Determine how long to wait before the next tick.
 * - If in a conversation where the other person spoke last: 2s (fast reply)
 * - Otherwise: 8s (normal exploration pace)
 */
function getTickDelay(agentId: string): number {
  const agent = useAgentsStore.getState().getAgent(agentId);
  if (!agent?.currentChatId) return 8000;

  const session = useChatsStore.getState().getAgentSession(agentId);
  if (!session || session.messages.length === 0) return 8000;

  const lastMsg = session.messages[session.messages.length - 1];
  if (lastMsg.id !== agentId) {
    // Someone else spoke last — we should respond quickly
    return 2000;
  }

  return 8000;
}

/**
 * When a message is sent in a chat, notify the other participants
 * to tick sooner so they can respond. This creates the back-and-forth flow.
 */
export function notifyChatPartners(chatId: string, speakerId: string): void {
  const session = useChatsStore.getState().sessions[chatId];
  if (!session) return;

  for (const pid of session.participants) {
    if (pid === speakerId) continue;
    const runtime = agentRuntimes.get(pid);
    if (!runtime || !runtime.running) continue;

    // Only schedule if we don't already have a fast tick pending
    if (!pendingFastTicks.has(pid)) {
      pendingFastTicks.add(pid);
      console.log(
        `[AI] ${pid}: Fast tick (responding to ${speakerId} in chat)`,
      );
      setTimeout(() => {
        pendingFastTicks.delete(pid);
        tickAgent(pid);
      }, 1500);
    }
  }
}

// --- Tick loop ---

async function tickAgent(agentId: string): Promise<void> {
  const runtime = agentRuntimes.get(agentId);
  if (!runtime || !runtime.running) return;

  // Don't tick if bridge isn't ready yet
  if (!bridgeFns) {
    console.log(`[AI] ${agentId}: Waiting for bridge...`);
    setTimeout(() => tickAgent(agentId), 2000);
    return;
  }

  try {
    // Build context/tools just before the call so dynamic state (region,
    // chat partners, points, etc.) is fresh when we actually hit the API,
    // not when we were originally queued behind the rate limiter.
    const result = await scheduleAgentCall(agentId, runtime.config.role, () => {
      const dynamicContext = buildDynamicContext(agentId, runtime);
      const tools = buildTools(agentId, runtime);

      console.log(
        `[AI] ${agentId}: Tick (${Object.keys(tools).length} tools, ${runtime.messages.length} msgs)`,
      );

      return generateText({
        model: getModel(runtime.config.role),
        system: runtime.systemPrompt + "\n\n" + dynamicContext,
        messages: runtime.messages,
        tools,
        // Bound the request so a cold-start hang can't wedge the agent
        // forever (see REQUEST_TIMEOUT_MS). Covers the whole multi-step
        // tool loop, not just the first step.
        abortSignal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        maxSteps: 5,
        // Larger than strictly needed for OpenAI/OpenRouter models, but
        // thinking/reasoning models (e.g. Qwen3.6 which emits an internal
        // `reasoning` channel before the user-visible content) easily use
        // 500-1500 tokens on reasoning alone.
        maxTokens: 4000,
        onStepFinish({ finishReason, toolCalls }) {
          if (toolCalls && toolCalls.length > 0) {
            for (const tc of toolCalls) {
              console.log(
                `[AI] ${agentId}: tool ${tc.toolName}(${JSON.stringify(tc.args)})`,
              );
            }
          } else if (finishReason === "stop" || finishReason === "length") {
            console.log(`[AI] ${agentId}: finished (${finishReason})`);
          }
        },
      });
    });

    // Append all response messages to history for continuity
    if (result.response?.messages) {
      runtime.messages.push(...result.response.messages);

      // Persist to the append-only log (never trimmed)
      const region = getAgentRegion(agentId);
      const now = Date.now();
      for (const msg of result.response.messages) {
        messageLog.push({
          agentId,
          agentName: runtime.config.name,
          agentRole: runtime.config.role,
          currentRegion: region,
          role: msg.role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
          timestamp: now,
        });
      }
    }

    // Trim history to prevent context overflow. Use safe trim to avoid
    // orphaning a `tool` message at index 0 (which would cause every
    // subsequent API call to 400 until the runtime is reset).
    if (runtime.messages.length > 40) {
      runtime.messages = safeTrimMessages(runtime.messages, 30);
    }

    // Check for hourly C-score snapshot
    checkHourlyCScoreSnapshot();

    // Show the agent's final text as a thought bubble
    if (result.text) {
      useAgentsStore.getState().updateThoughtBubble(agentId, {
        content: result.text,
        timestamp: Date.now(),
        duration: 10000,
      });
    }

    // Log step summary
    const toolCallCount =
      result.steps?.reduce((sum, s) => sum + (s.toolCalls?.length ?? 0), 0) ??
      0;
    if (toolCallCount > 0 || result.text) {
      console.log(
        `[AI] ${agentId}: Completed (${result.steps?.length ?? 0} steps, ${toolCallCount} tool calls)${result.text ? ` - "${result.text.slice(0, 80)}..."` : ""}`,
      );
    }

    // --- Text-to-tool-call fallback for reasoning models ---
    //
    // Some models (notably Qwen3-family thinking models) occasionally
    // describe the action they want to take in plain text instead of
    // emitting the corresponding tool call, e.g. the response text is
    // `"say: Hello there."` with zero tool_calls. If the agent is in a
    // chat and no `say` tool was called this tick, recover the message
    // from the text and send it as a chat message. Prevents dropped
    // utterances without requiring a re-prompt.
    const recoveredSayFromText = recoverSayFromText({
      agentId,
      agentName: runtime.config.name,
      text: result.text,
      toolCallsThisTick: result.steps?.flatMap((s) => s.toolCalls ?? []) ?? [],
    });
    if (recoveredSayFromText) {
      console.log(
        `[AI] ${agentId}: Recovered say from text: "${recoveredSayFromText.slice(0, 80)}${recoveredSayFromText.length > 80 ? "..." : ""}"`,
      );
    }

    // Schedule next tick — faster if in an active conversation waiting for our reply
    const nextDelay = getTickDelay(agentId);
    setTimeout(() => tickAgent(agentId), nextDelay);
  } catch (error: unknown) {
    const err = error as {
      name?: string;
      status?: number;
      statusCode?: number;
      message?: string;
      data?: unknown;
      responseBody?: string;
    };
    const status = err?.status ?? err?.statusCode;
    const is429 = status === 429 || err?.message?.includes("429");
    const is400 =
      status === 400 ||
      err?.message?.includes("400") ||
      err?.responseBody?.includes("tool") ||
      err?.message?.toLowerCase().includes("tool_call");

    // Model emitted a tool call with invalid/missing arguments (e.g.
    // Qwen3.6 sometimes calls `say({})` with no `message`). This is a
    // soft error — no corruption, no rate limiting — just a model
    // hiccup. Skip this tick quickly without the full stack trace.
    const isInvalidToolArgs =
      err?.name === "AI_InvalidToolArgumentsError" ||
      (err?.message?.includes("Invalid arguments for tool") &&
        err?.message?.includes("Type validation failed"));

    if (isInvalidToolArgs) {
      // Extract just the tool name from the error message for a clean log.
      const toolMatch = err?.message?.match(/tool\s+([a-z_]+):/i);
      const toolName = toolMatch ? toolMatch[1] : "unknown";
      console.warn(
        `[AI] ${agentId}: Model emitted malformed ${toolName}() call (missing required args), skipping tick`,
      );
      setTimeout(() => tickAgent(agentId), 2000);
      return;
    }

    // Self-heal: if the message history is corrupted (starts with an
    // orphan `tool` message) OR we got a 400 that smells like a
    // tool-call / tool-result mismatch, reset the runtime to its initial
    // state so the agent can recover instead of looping on the same
    // broken request forever.
    const runtime = agentRuntimes.get(agentId);
    const corrupted = runtime && isMessageHistoryCorrupted(runtime.messages);
    if (runtime && (corrupted || is400)) {
      console.warn(
        `[AI] ${agentId}: Resetting message history ${corrupted ? "(orphan tool at index 0)" : "(400 — likely tool-call/result mismatch)"}`,
      );
      runtime.messages = [{ role: "user", content: INITIAL_USER_MESSAGE }];
    }

    const backoff = is429 ? 30000 : 5000;
    const reason = is429
      ? "429 rate limited"
      : is400
        ? `400 ${err?.message ?? ""}`.trim()
        : (err?.message ?? "unknown");
    console.warn(
      `[AI] ${agentId}: Tick failed (${reason}), retry in ${backoff / 1000}s`,
    );
    if (!is429) console.error("[AI] Full error:", error);
    setTimeout(() => tickAgent(agentId), backoff);
  }
}

// --- Public API ---

/** Initialize all agents and start their tick loops. */
export function initAgents(agents: AgentConfig[]): void {
  console.log(`[AI] Initializing ${agents.length} agents...`);

  agents.forEach((config, index) => {
    const runtime: AgentRuntime = {
      config,
      systemPrompt: buildSystemPrompt(config),
      messages: [{ role: "user", content: INITIAL_USER_MESSAGE }],
      memoryStore: new MemoryStore(),
      relationshipState: new RelationshipState(),
      running: true,
    };

    agentRuntimes.set(config.id, runtime);

    // Persist the initial message to the log
    messageLog.push({
      agentId: config.id,
      agentName: config.name,
      agentRole: config.role,
      currentRegion: "unknown", // bridge not ready yet at init time
      role: "user",
      content: INITIAL_USER_MESSAGE,
      timestamp: Date.now(),
    });

    // Stagger initial starts so we don't flood the API
    const delay = 3000 + index * 2000;
    console.log(
      `[AI] ${config.id} (${config.name}): First tick in ${delay / 1000}s`,
    );
    setTimeout(() => tickAgent(config.id), delay);
  });
}

/** Stop all agent tick loops. */
export function stopAllAgents(): void {
  for (const runtime of agentRuntimes.values()) {
    runtime.running = false;
  }
}

/** Get the total number of messages across all agents. */
export function getTotalMessages(): number {
  let total = 0;
  for (const runtime of agentRuntimes.values()) {
    total += runtime.messages.length;
  }
  return total;
}

/** Determine which region an agent is currently in based on their world position. */
function getAgentRegion(agentId: string): string {
  if (!bridgeFns) return "unknown";

  // Use world-space coordinates from the Phaser sprite (not screen coords from Zustand)
  const worldPos = bridgeFns.getAgentWorldPosition(agentId);
  if (!worldPos) return "unknown";

  const regions = bridgeFns.getRegions();
  for (const region of regions) {
    if (
      worldPos.x >= region.x &&
      worldPos.x <= region.x + region.width &&
      worldPos.y >= region.y &&
      worldPos.y <= region.y + region.height
    ) {
      return region.label;
    }
  }
  return "unknown";
}

/** Export all agent messages as JSONL for analysis. */
export function exportMessagesAsJSONL(): string {
  const allLines: Array<Record<string, unknown>> = [];
  const agentsStore = useAgentsStore.getState();

  // 1. All LLM messages from the persistent log (complete history, never trimmed)
  for (const entry of messageLog) {
    allLines.push({ ...entry });
  }

  // 2. All chat messages
  for (const session of useChatsStore.getState().getAllSessions()) {
    for (const msg of session.messages) {
      const agentRegion = getAgentRegion(msg.id);
      allLines.push({
        agentId: msg.id,
        agentName: msg.name,
        currentRegion: agentRegion,
        role: "chat",
        content: msg.content,
        timestamp: msg.timestamp,
        chatId: session.id,
        chatParticipants: session.participants.map(
          (pid) => agentsStore.getAgent(pid)?.name ?? pid,
        ),
        c_score: msg.cScores ?? {},
      });
    }
  }

  // 3. Hourly C-score snapshots
  for (const snapshot of cScoreSnapshots) {
    allLines.push({
      role: "cscore_snapshot",
      simulationTime: snapshot.simulationTime,
      timestamp: snapshot.realTimestamp,
      scores: snapshot.scores,
    });
  }

  // 4. Final C-score snapshot at download time
  const simTime = getCurrentGameTime();
  const prisoners = agentsStore
    .getAllAgents()
    .filter((a) => a.role === "prisoner");

  if (simTime) {
    const hours = simTime.getHours();
    const minutes = simTime.getMinutes().toString().padStart(2, "0");
    const ampm = hours >= 12 ? "PM" : "AM";
    const h12 = hours % 12 || 12;

    allLines.push({
      role: "cscore_snapshot",
      simulationTime: `${h12}:${minutes} ${ampm} (at download)`,
      timestamp: Date.now(),
      scores: prisoners.map((p) => ({
        id: p.id,
        name: p.name,
        points: p.points,
        region: getAgentRegion(p.id),
      })),
    });
  }

  return allLines.map((m) => JSON.stringify(m)).join("\n");
}
