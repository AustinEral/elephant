# Failure Analysis: reflect-hierarchical-logging-2

**Run**: `reflect-hierarchical-logging-2` | **Accuracy**: 83.8% (129/154) | **25 failures**
**Artifact**: `bench/locomo/results/archive/legacy-v0/reflect-hierarchical-logging-2.json`

## Root Cause Summary

| Root Cause | Count | % of Failures |
|---|---|---|
| **A) Retain gap** — detail never extracted/stored | **14** | 56% |
| **C) Synthesis gap** — facts retrieved, LLM answered wrong | **5** | 20% |
| **D) Exhaustion** — ran out of iterations, empty answer | **4** | 16% |
| **A+C) Retain gap + Synthesis gap** | **2** | 8% |

## Category Breakdown

| Category | Failures | Retain Gap | Synthesis Gap | Exhaustion | Mixed |
|---|---|---|---|---|---|
| multi-hop (37 total) | 3 | 1 | 2 | 0 | 0 |
| single-hop (32 total) | 4 | 1 | 1 | 1 | 1 |
| temporal (13 total) | 1 | 0 | 0 | 0 | 1 |
| open-domain (70 total) | 17 | 12 | 2 | 3 | 0 |

## Detailed Failure Table

### D) Exhaustion — 4 failures (EMPTY answers)

All 4 hit `reflect agent exhausted all iterations without calling done()`. **BUG**: On iteration 7 (`forced_tool="done"`), LLM ignores the tool constraint and calls `search_observations`/`recall` instead. The code provides `done_only` tools with `ToolChoice::Required`, but the LLM doesn't comply, and there's no fallback.

| # | QID | Category | Question | GT | Root Issue |
|---|---|---|---|---|---|
| 8 | 73b759 | single-hop | What book did Melanie read from Caroline's suggestion? | "Becoming Nicole" | Facts exist separately ("Caroline recommends Becoming Nicole" + "Melanie reading book Caroline recommended") but extraction never linked them. LLM searched 8 iters, never inferred the connection. Also exhaustion bug. |
| 10 | a133d4 | open-domain | What is Melanie's hand-painted bowl a reminder of? | art and self-expression | "art and self-expression" NOT in DB for the bowl. Bowl described as "gift for 18th birthday" only. Retain gap + exhaustion. |
| 20 | 5281df | open-domain | What painting did Melanie show to Caroline on October 13, 2023? | A painting inspired by sunsets with a pink sky | No facts mention Oct 13 sharing or "pink sky." Sunset painting exists but not linked to Oct 13 sharing event. Retain gap + exhaustion. |
| 22 | e15b99 | open-domain | What kind of painting did Caroline share with Melanie on October 13, 2023? | An abstract painting with blue streaks on a wall | "Abstract painting features blue streaks" exists for Melanie, not Caroline. No Oct 13 sharing event stored. Retain gap + exhaustion. |

### A) Retain Gap — 14 failures (detail never extracted)

| # | QID | Category | Question | GT | What's Missing |
|---|---|---|---|---|---|
| 2 | f7e9f3 | single-hop | What books has Melanie read? | "Nothing is Impossible", "Charlotte's Web" | "Nothing is Impossible" not in DB. Only Charlotte's Web. |
| 4 | fb1e0a | single-hop | What types of pottery have Melanie and her kids made? | bowls, cup | "cup" not in DB. Only "pots", "bowls", "plate". |
| 9 | 0c9c91 | open-domain | What did Melanie realize after the charity race? | self-care is important | "self-care" not in DB. Only "rewarding and thought-provoking." |
| 11 | 84f0bd | open-domain | What kind of pot did Mel and her kids make with clay? | a cup with a dog face on it | "cup with dog face" not in DB. Only "each made their own pots." |
| 12 | 64d028 | open-domain | What did Mel and her kids paint in their latest project in July 2023? | a sunset with a palm tree | "sunset with a palm tree" not in DB. Only "nature-inspired painting." |
| 13 | f0f797 | open-domain | What did Caroline see at the council meeting for adoption? | many people wanting to create loving homes for children in need | Not in this bank. Only "inspiring and emotional." |
| 14 | 119 | open-domain | How did Melanie feel while watching the meteor shower? | in awe of the universe | "awe" not in DB. Only "made wishes while watching the sky light up." |
| 16 | afb51a | open-domain | What precautionary sign did Melanie see at the café? | A sign stating that someone is not being able to leave | Sign content not extracted. Only "thoughtful signs, which she described as a precaution." |
| 18 | 9aa6a6 | open-domain | What does Melanie do to keep herself busy during her pottery break? | Read a book and paint | "Read a book" during pottery break not linked. Painting exists, reading doesn't mention break context. |
| 19 | ebdd58 | open-domain | What did the posters at the poetry reading say? | "Trans Lives Matter" | "Trans Lives Matter" not in DB at all. Only that it was "empowering" and about "self-expression." |
| 21 | 988b61 | open-domain | How do Melanie and Caroline describe their journey together? | An ongoing adventure of learning and growing | Fact exists as Caroline's opinion but not as their shared description. Recall may not have surfaced it for this question framing. |
| 23 | c380bb | open-domain | How did Melanie feel after the accident? | Grateful and thankful for her family | "Grateful" not stored for Melanie in accident context. Only children's feelings recorded. |
| 24 | 1f3680 | open-domain | What was Melanie's reaction to her children enjoying the Grand Canyon? | She was happy and thankful | No emotion recorded for Grand Canyon. Only "enjoyed" it. |
| 25 | 3f4a3d | open-domain | What did Melanie do after the road trip to relax? | Went on a nature walk or hike | "Nature walk" or "hike after road trip" not in DB. Camping trip exists but is a different activity. |

### C) Synthesis Gap — 5 failures (facts exist, LLM answered wrong)

| # | QID | Category | Question | GT | What Went Wrong |
|---|---|---|---|---|---|
| 1 | e462ca | multi-hop | When is Melanie planning on going camping? | June 2023 | Fact "considering camping in June 2023" EXISTS. LLM retrieved Oct camping trip instead and answered Oct 2023. |
| 5 | 9b5f7d | single-hop | What symbols are important to Caroline? | Rainbow flag, transgender symbol | Rainbow flag facts exist (mural, sidewalk). LLM answered with grandmother's necklace instead. Wrong interpretation of "symbols." |
| 6 | 8fe09f | multi-hop | When is Caroline's youth center putting on a talent show? | September 2023 | Fact says "next month" with temporal_start=2023-09-01. LLM echoed "next month" without resolving to September. |
| 15 | e869da | open-domain | Why did Melanie choose to use colors and patterns in her pottery? | She wanted to catch the eye and make people smile | "Eye-catching" exists but "make people smile" not stored. Partial: LLM got "eye-catching" right but missed "smile." |
| 17 | c502a7 | open-domain | What setback did Melanie face in October 2023? | She got hurt and had to take a break from pottery | Injury fact EXISTS but dated September 2023. LLM answered with son's October road trip accident instead. Temporal mismatch. |

### A+C) Mixed — 2 failures

| # | QID | Category | Question | GT | What Went Wrong |
|---|---|---|---|---|---|
| 3 | 9a7616 | multi-hop | When did Melanie run a charity race? | The Sunday before 25 May 2023 | Extraction stored "Saturday, 20 May 2023" — wrong day. GT is Sunday, May 21. **Retain gap** (wrong date). |
| 7 | 1d2c57 | temporal | What personality traits might Melanie say Caroline has? | Thoughtful, authentic, driven | "Thoughtful" exists, "authentically" exists (as verb, not trait), "driven" missing. **Retain gap** (missing driven) + **Synthesis gap** (didn't map "live authentically" → "authentic"). |

## Exhaustion Bug Detail

The forced-done mechanism on the last iteration doesn't work. On iteration 7:
- Code sets `iter_tools = &done_only` (only the `done` tool) and `tool_choice = Some(ToolChoice::Required)`
- LLM ignores this and calls `search_observations` or `recall` anyway
- Code processes the non-done tool call (recall branch handles it), appends messages
- Loop ends, falls through to exhaustion error

**Root cause**: The LLM provider (Anthropic API) is being asked to use a required tool, but the model still returns a different tool name not in the provided list. The code doesn't validate that the returned tool name matches the allowed tools.

## Prioritized Fixes

### Fix 1: Exhaustion fallback (4 failures → 4 recoverable)
When the reflect agent exhausts iterations without `done`, instead of returning an error:
1. On the last iteration, if LLM doesn't call `done`, extract any text content from the response as the answer
2. If no text content, synthesize from accumulated facts with a simple prompt
3. Additionally: validate that returned tool calls match the provided tool list; if not, treat as text response

### Fix 2: Improve fact extraction for specific details (14 retain gaps → up to 14 recoverable)
The extraction prompt misses specific details that open-domain questions ask about:
- Emotions/feelings ("in awe", "grateful", "happy and thankful")
- Specific visual details ("sunset with palm tree", "cup with dog face", "blue streaks on wall", "pink sky")
- Specific text on signs/posters ("Trans Lives Matter", café sign content)
- Specific realizations ("self-care is important")

**Pattern**: The extraction captures events and high-level summaries but drops the *specific wording* and *emotional nuance* from conversations. The LoCoMo ground truths often use exact phrases from the source conversations.

**Proposed extraction prompt change**: Add guidance to extract:
- Exact phrases used to describe feelings and realizations
- Specific visual details of art, objects, and places
- Text content of signs, posters, and written materials
- Character reactions and emotional responses, not just the events

### Fix 3: Temporal resolution in synthesis (2 failures)
For #6 (talent show "next month") and #17 (injury date mismatch), the LLM needs to resolve relative dates to absolute dates using temporal_start metadata. Currently the temporal context is prepended for reranking but may not be visible to the reflect agent.

### Fix 4: Recall relevance for multi-hop questions (2 failures)
For #1 (camping June vs October) and #5 (symbols), the right facts exist but the wrong ones were prioritized. This may be a reranker issue — more recent/detailed facts outranking the specific answer.

## Impact Estimate

| Fix | Failures Addressed | Estimated Recovery | Effort |
|---|---|---|---|
| Fix 1: Exhaustion fallback | 4 | 2-3 (some are also retain gaps) | Small — code change in reflect loop |
| Fix 2: Extraction prompt | 14 | 7-10 (some details may be too specific) | Medium — prompt tuning + re-run retain |
| Fix 3: Temporal resolution | 2 | 1-2 | Small — prompt/context change |
| Fix 4: Recall relevance | 2 | 1 | Medium — reranker tuning |

**Total potential**: Recovering 11-16 of 25 failures → accuracy from 83.8% to ~91-94%.
