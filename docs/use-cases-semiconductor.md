# Use Cases: Semiconductor Industry

How triz-ai can be applied across functions in a semiconductor company, beyond traditional R&D.

## Engineering & R&D

### Problem Solving During Development

Engineers use triz-ai to resolve technical contradictions encountered during chip and module design.

**Example:** "Increase SiC MOSFET switching speed without increasing EMI"

```bash
triz-ai analyze "Increase SiC MOSFET switching speed without increasing EMI" --deep
```

**Value:** Structured problem decomposition with patent-backed solution directions, rather than trial-and-error iteration.

### Physical Contradiction Resolution

Design trade-offs where a component must satisfy opposing requirements.

**Example:** "The solder joint must be rigid for mechanical reliability but flexible to absorb thermal cycling stress"

```bash
triz-ai analyze "The solder joint must be rigid for reliability but flexible for thermal cycling"
```

**Value:** Recommends separation principles (time, space, scale, condition) with real patent evidence showing how others solved similar contradictions.

### Cost Reduction and Simplification

Trimming analysis identifies components that can be removed while redistributing their functions.

**Example:** "Reduce the BOM cost of this gate driver circuit"

```bash
triz-ai analyze "Reduce the BOM cost of this gate driver circuit" --method trimming
```

**Value:** Systematic approach to cost reduction grounded in TRIZ trimming methodology, not just "use cheaper parts."

### Technology Roadmapping

Trends analysis positions a technology on TRIZ evolution curves and predicts next stages.

**Example:** "What is the next generation of SiC packaging technology?"

```bash
triz-ai analyze "What is the next generation of SiC packaging technology?" --method trends
```

**Value:** Patent-grounded technology forecasting for roadmap planning.

---

## Field Application Engineering (FAE Support)

### Real-Time Customer Problem Solving

FAEs are the front line between the company and design-in customers. When a customer describes a design challenge, an FAE can generate structured solution directions on the spot.

**Example workflow:**
1. Customer: "We're seeing thermal runaway in parallel IGBT modules under high di/dt conditions"
2. FAE runs: `triz-ai analyze "Prevent thermal runaway in parallel IGBT modules under high di/dt" --deep`
3. Gets: contradiction analysis, IFR, patent examples from competitors and peers, concrete solution directions
4. Discusses solutions with customer in the same meeting

**Value:** Faster design-in cycles. Problems that previously required R&D escalation (days/weeks) can be addressed in a single customer meeting.

### Application Note Discovery

FAEs encounter recurring customer problems in specific application domains. triz-ai can identify patterns and underused principles.

```bash
triz-ai discover --domain "motor drive"
```

**Value:** Identifies which TRIZ principles are underused in a domain — these are potential topics for new application notes or reference designs.

---

## Technical Marketing

### Content Generation for White Papers and App Notes

Technical marketing teams produce content explaining how to solve design problems. triz-ai provides structured material:

- Problem framed as a contradiction (engineers relate to this framing)
- Ideal Final Result (a crisp articulation of the perfect outcome)
- Solution directions backed by real patents with assignees and filing dates
- Prior art landscape showing the state of the art

**Example:** Writing an app note on EMI mitigation in power converters:

```bash
triz-ai analyze "Reduce conducted EMI in high-frequency power converters without adding filter stages" --deep
```

**Value:** Content grounded in patent evidence and structured TRIZ methodology, not generic claims. More credible and technically rigorous than typical marketing material.

### Technology Differentiation Messaging

Marketing needs to articulate *why* a company's technology is better. triz-ai can identify the specific technical contradictions that products solve.

**Example:** Analyzing what contradictions a new product generation resolves:

```bash
triz-ai analyze "Achieve lower Rds_on in a smaller die area without degrading avalanche ruggedness"
```

**Value:** Translates product specs into problem-solution narratives that resonate with engineering audiences. "We solved X contradiction" is more compelling than "we have lower Rds(on)."

### Trade Show and Workshop Preparation

Preparing technical demos and presentations with concrete, patent-backed examples.

```bash
triz-ai analyze "How to detect delamination in power module packages without adding sensors" --method su-field
```

**Value:** Live demonstration material showing AI-powered problem solving, positioning the company as an innovation leader.

---

## Competitive Intelligence

### Patent Landscape Analysis

Ingest competitor patents to understand their innovation focus and identify gaps.

```bash
# Ingest competitor patent data
triz-ai ingest data/patents/competitor_ti_power.json
triz-ai ingest data/patents/competitor_stmicro_sic.json

# Discover which principles competitors rely on
triz-ai discover --domain "SiC power module"

# See matrix statistics — where is innovation clustering?
triz-ai matrix stats
```

**Value:** Data-driven view of where competitors are investing R&D, which contradictions they've solved, and where gaps remain.

### Competitive Displacement

Identify technical limitations in competitor approaches by analyzing what contradictions their solutions do NOT address.

**Example:** If a competitor's GaN solution optimizes for efficiency but creates thermal management challenges:

```bash
triz-ai analyze "GaN HEMT thermal management in high power density applications"
```

**Value:** Arms sales teams with specific technical arguments for displacement, backed by patent evidence.

### Emerging Technology Monitoring

Track how innovation is evolving in specific domains using the evolution pipeline.

```bash
triz-ai evolve
triz-ai evolve --parameters
```

**Value:** Identifies candidate new engineering parameters and principles emerging from recent patents — early signals of where the industry is heading.

---

## Sales Enablement

### Customer Workshop Facilitation

Run triz-ai-powered problem-solving workshops with key accounts. The structured approach positions the company as a problem-solving partner, not just a component supplier.

**Workshop flow:**
1. Customer describes their design challenge
2. triz-ai classifies the problem type and formulates the Ideal Final Result
3. Patent search surfaces relevant prior art from the domain
4. Solution directions are generated and discussed
5. Follow-up: which solutions map to the company's product portfolio

**Value:** Transforms the customer relationship from transactional ("buy our chip") to consultative ("we help you solve your hardest problems").

### Design-In Acceleration

Sales teams can quantify the value of faster problem resolution:

- FAE resolves customer issue in 1 meeting instead of 3
- Fewer R&D escalations per quarter
- Shorter time from customer inquiry to design-in commitment

### Solution Selling Narratives

Sales presentations structured around problems and contradictions are more compelling than feature/spec comparisons.

**Instead of:** "Our IGBT has 15% lower switching losses"
**Say:** "Your application faces a contradiction between switching frequency and thermal performance. Here's how our IGBT resolves it — and here are patents showing the approach works."

---

## Product Planning & Strategy

### Innovation Gap Analysis

Compare the company's patent portfolio against the full contradiction matrix to find cells with no coverage.

```bash
triz-ai matrix stats
```

**Value:** Identifies parameter combinations where the company has no IP — potential areas for strategic R&D investment or acquisition.

### New Product Justification

Use patent trend data to build evidence-based business cases for new products.

**Example:** "Patents in automotive SiC show a trend toward [evolution stage X]. Our current portfolio covers stages 1-3 but not 4. Investing in [technology Y] captures the next wave."

**Value:** Product proposals grounded in patent evidence, not just market surveys.

### M&A and Partnership Intelligence

Patent landscape analysis reveals which companies hold IP in areas of strategic interest.

```bash
triz-ai discover --domain "wide bandgap packaging"
```

**Value:** Identifies potential acquisition targets or partnership candidates based on complementary IP portfolios.

---

## Deployment Scenario: Web Search Only (No Patent Database)

triz-ai delivers full value with zero data infrastructure by replacing the patent database with a live web search research tool. This is the recommended starting point for internal rollout — immediate value, no setup, no data pipeline to maintain.

### How It Works

The TRIZ engine (contradiction analysis, IFR, 40 principles, matrix lookup, solution directions) runs entirely from the LLM and bundled TRIZ knowledge. A web search tool plugs into the research tool interface to provide real-time grounding:

```python
from triz_ai import ResearchTool
from triz_ai.engine.router import route
from triz_ai.llm.client import LLMClient

web_search = ResearchTool(
    name="web_search",
    description="Search the web for technical solutions, datasheets, and app notes.",
    fn=my_search_fn,  # wraps Tavily, SerpAPI, Google, or Bing
    stages=["context", "search", "enrichment"],
)

result = route(
    "Reduce EMI in GaN half-bridge without adding filter stages",
    LLMClient(),
    research_tools=[web_search],
)
```

No database initialization, no patent ingestion, no `triz-ai init`. Just an API key for the LLM and a web search provider.

### What Web Search Provides vs. Patents

| Dimension | Patent database | Web search only |
|-----------|----------------|-----------------|
| **Setup time** | Hours (curate, ingest, classify) | Minutes (API key only) |
| **Freshness** | 18-month patent publication lag | Real-time |
| **Source breadth** | Inventions only | Datasheets, app notes, forums, papers, teardowns |
| **Depth on IP** | Deep (claims, classifications, assignees) | Shallow (snippets, links) |
| **Maintenance** | Ongoing ingestion pipeline | Zero |
| **Offline use** | Yes | No (requires internet) |

The patent database becomes a later enhancement for teams that want deep IP analysis, competitive patent landscaping, or offline operation.

### Target Audience

#### Primary: Field Application Engineers (FAEs)

FAEs are the highest-impact, lowest-friction audience for web-search-only deployment.

- **Profile:** Engineers with 5-15 years domain expertise, customer-facing, time-pressured, solving 3-5 customer design problems per week
- **Pain point:** Escalating to R&D for every hard question adds days/weeks to design-in timelines
- **How they'd use it:** Run `triz-ai analyze` during or before customer meetings; get structured solution directions enriched with current web results (datasheets, app notes, forum solutions)
- **Adoption path:** Share as a CLI tool or wrap in a simple internal web UI; FAEs already use terminal tools and scripts
- **Success metric:** Reduction in R&D escalations per quarter, shorter time-to-design-in

#### Secondary: Application Engineers in R&D

Engineers writing app notes, reference designs, and evaluation board documentation.

- **Profile:** Deep technical specialists, produce content consumed by customers and FAEs
- **Pain point:** Researching state-of-art before writing an app note is manual and time-consuming
- **How they'd use it:** `--deep` mode to get a comprehensive analysis with web-enriched context before starting a document; use the contradiction framing and solution directions as the app note structure
- **Success metric:** Faster app note production cycle, higher technical quality

#### Tertiary: Technical Marketing Managers

Non-daily users who leverage outputs for positioning and messaging.

- **Profile:** Engineering background, now focused on market positioning and content strategy
- **Pain point:** Translating product specs into compelling technical narratives
- **How they'd use it:** Request analyses from FAEs or application engineers; use IFR and contradiction framing in presentations and white papers
- **Success metric:** More technically grounded marketing materials, better reception at design-in reviews

### Use Cases for Web-Search-Only Deployment

#### 1. FAE Problem Solving Copilot

The primary daily-use scenario. An FAE receives a customer question and gets a structured response in minutes.

**Workflow:**
1. Customer emails: "We're seeing ringing on the gate drive signal at turn-on, causing shoot-through in our half-bridge"
2. FAE runs: `triz-ai analyze "Eliminate gate drive ringing at turn-on without slowing switching speed in half-bridge topology" --deep`
3. Web search tool pulls: relevant TI/Infineon/STMicro app notes, EEVblog forum threads, recent IEEE papers
4. triz-ai outputs: contradiction analysis (speed vs. ringing = Principle 21: Rushing Through, Principle 28: Mechanics Substitution), IFR, solution directions grounded in web results
5. FAE responds to customer with structured analysis and recommended approach — same day

#### 2. Pre-Meeting Preparation

Before a customer design review, FAEs prepare by analyzing the customer's known design challenges.

**Workflow:**
1. FAE knows the customer is designing a 800V SiC inverter for EV traction
2. Runs several analyses on anticipated challenges: thermal management, EMI, short-circuit protection
3. Arrives at the meeting with structured solution directions ready
4. Positions the company as a knowledgeable partner, not just a vendor presenting datasheets

#### 3. Competitive Technical Comparison

When a customer is evaluating a competitor's solution, use triz-ai to identify technical trade-offs.

**Workflow:**
1. Customer says: "Competitor X claims their GaN solution doesn't need a heatsink"
2. FAE runs: `triz-ai analyze "Achieve adequate thermal dissipation in GaN power stage without heatsink"`
3. Web search pulls: competitor datasheets, thermal analysis papers, teardown reports
4. triz-ai identifies the underlying contradiction and what trade-offs the competitor likely made
5. FAE presents: "Their approach resolves thermal management through [principle], but creates [secondary contradiction]. Our solution addresses both."

#### 4. Quick Technical Content Drafts

Application engineers use triz-ai output as a starting structure for technical documents.

**Workflow:**
1. Assignment: Write an app note on "EMI mitigation techniques for high-frequency DC-DC converters"
2. Run: `triz-ai analyze "Reduce conducted EMI in high-frequency DC-DC converters without adding filter stages or reducing switching frequency" --deep`
3. Output provides: problem framing (contradiction), state-of-art (web search), IFR, solution directions
4. Engineer uses this as the skeleton of the app note, adding measurement data and specific product recommendations

#### 5. Training and Onboarding

New engineers learn TRIZ methodology and semiconductor domain knowledge simultaneously.

**Workflow:**
1. New hire receives a set of canonical problems: "Analyze these 10 common power electronics design challenges using triz-ai"
2. Each analysis teaches: how to frame problems as contradictions, what TRIZ principles apply, what the state of the art looks like
3. Builds both TRIZ intuition and domain knowledge faster than reading textbooks

### Branding and Internal Promotion

#### Positioning Statement

> **"Your AI engineering copilot that thinks in contradictions"**
>
> triz-ai helps engineers solve the hardest design trade-offs — not by choosing between A and B, but by finding solutions that deliver both. Powered by 80 years of TRIZ methodology, grounded in real-time technical knowledge.

#### Internal Brand Name Options

For internal rollout, a simpler name may resonate better than "triz-ai":

| Name | Angle | Pro | Con |
|------|-------|-----|-----|
| **TRIZ Copilot** | Familiar AI framing | Engineers understand "copilot" | Sounds like another chatbot |
| **Contradiction Solver** | Describes exactly what it does | Self-explanatory | Narrow perception |
| **IFR Engine** | TRIZ-native naming (Ideal Final Result) | Appeals to TRIZ-trained engineers | Obscure to newcomers |
| **InnoSolve** | Innovation + Solve | Broad appeal, corporate-friendly | Generic |
| **Patent-free mode: Design Assist** | Emphasizes zero-setup value | Lowers adoption barrier | Undersells the TRIZ depth |

Recommendation: **"TRIZ Copilot"** for the tool itself, with the tagline **"Think in contradictions"** — it bridges the familiar (AI copilot) with the distinctive (TRIZ methodology).

#### Promotion Channels (Internal)

| Channel | Action | Timing |
|---------|--------|--------|
| **FAE team meeting** | Live demo: take a real customer problem from last week, run it through triz-ai, compare output to what was actually done | Week 1 |
| **Application engineering newsletter** | "Tool of the month" feature with 2-3 example analyses relevant to current projects | Week 2 |
| **Internal tech talk / brown bag** | 30-min session: "How AI + TRIZ solved [specific Infineon problem]" | Month 1 |
| **Confluence/SharePoint page** | Setup guide, example gallery, FAQ | Week 1 (alongside demo) |
| **Slack/Teams channel** | `#triz-copilot` for sharing analyses, tips, and feature requests | Week 1 |
| **Quarterly innovation review** | Present adoption metrics: analyses run, R&D escalations avoided, design-in impact | Quarter 1 |

#### Demo Script (5 minutes)

A proven sequence for internal demos:

1. **Hook (30s):** "Last week, [FAE name] spent 3 days getting an answer for a customer about [problem]. Let me show you how long it takes now."
2. **Live run (90s):** Type the problem, run `--deep`, show output streaming
3. **Walk through output (120s):** Point out: contradiction identified, IFR formulated, web results pulled in, solution directions with specific techniques
4. **Compare (30s):** "Here's what we actually recommended to the customer — triz-ai got to the same answer, plus two alternatives we didn't consider"
5. **Call to action (30s):** "Try it on your next customer problem. Here's the setup guide."

#### Success Metrics for Rollout

| Metric | Target (Q1) | How to measure |
|--------|-------------|----------------|
| Active users (monthly) | 15-20 FAEs | CLI usage telemetry or self-reported |
| Analyses per week | 30+ | Log aggregation |
| R&D escalations avoided | 5+ per month | FAE self-reporting |
| Time-to-first-response for customer issues | 30% reduction | Compare before/after averages |
| Internal NPS / satisfaction | >7/10 | Quick survey at 90 days |

### Growth Path: Web Search → Patent Database → Full Platform

The web-search-only deployment is phase 1 of a natural progression:

```
Phase 1: Web Search Only (Month 1-3)
  → Zero setup, immediate value
  → Proves the concept with FAEs and app engineers
  → Collects real usage data on which problems are analyzed most

Phase 2: Add Patent Database (Month 3-6)
  → Ingest company + competitor patents for target domains
  → Enables IP landscape analysis and competitive displacement use cases
  → Patent data supplements (not replaces) web search

Phase 3: Custom Research Tools (Month 6+)
  → Plug in internal data sources: test databases, simulation results, failure analysis reports
  → Connect to BigQuery for large-scale patent analytics
  → Build domain-specific tools (e.g., SPICE simulation lookup, reliability database)
```

Each phase adds value without disrupting what's already working. Users who adopted in Phase 1 keep their workflow — they just get richer results.

---

## Where triz-ai Does NOT Fit

For clarity, these are areas outside the tool's scope:

- **Brand marketing, advertising, demand generation campaigns** — no technical contradiction to solve
- **Pricing, deal structuring, CRM workflows** — wrong tool entirely
- **Non-technical sales** (distribution, logistics, channel management) — no engineering problem involved
- **Financial analysis, market sizing** — triz-ai analyzes technical problems, not markets
- **HR, legal, compliance** — unrelated domain

---

## Summary by Function

| Function | Primary use case | Impact |
|----------|-----------------|--------|
| R&D | Contradiction resolution, cost reduction, roadmapping | Faster innovation cycles |
| FAE | Real-time customer problem solving | Shorter design-in cycles |
| Technical Marketing | Content generation, differentiation messaging | More credible, structured content |
| Competitive Intelligence | Patent landscape, gap analysis | Data-driven competitive strategy |
| Sales | Customer workshops, solution selling | Consultative positioning |
| Product Planning | Innovation gaps, trend analysis, M&A intel | Evidence-based investment decisions |
