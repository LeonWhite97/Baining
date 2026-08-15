# PIS-IN AOI V3.5 Documentation and PDF Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or execute each task inline with document/PDF verification. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a consistent V3.5 project document set and visually verified PDFs that match the runnable application and corrected metric targets.

**Architecture:** Markdown is the editable source of truth. A new version-controlled ReportLab builder generates customer-facing and interview-facing PDFs from approved content and screenshots captured from the running React app. Internal-only budget details stay in the project master document and interview evidence package, never in frontend screenshots or customer PDFs.

**Tech Stack:** Markdown, Mermaid, ReportLab, pypdf, Poppler, Playwright screenshots, `tools/docs/build_v35_pdf.py`.

## Global Constraints

- Replace old 42% -> 9% claims with staged targets: baseline 12%, PoC <=6%, controlled rollout <=3%, mature <=1.5%.
- Label the denominator as the original AOI NG candidate pool.
- Keep the separate full-inspection false-positive target at <=0.5% and never mix its denominator with the AOI NG candidate pool.
- Keep historical new-product-cycle and inference figures separate from field targets.
- Do not expose the RMB 5 million budget in frontend or customer-facing PDFs.
- Render and inspect every final PDF page before delivery.
- Preserve the V3.5 Source Key, Light ID attachment, multi-model result, quarantine, and rollback design.

---

### Task 1: Metric and Claim Consistency Audit

**Files:**
- Modify: `技术方案设计书.md`
- Modify: `PIS-IN_AOI_AI智能质检_简历项目描述与面试讲解.md`
- Modify: `PIS-IN_AOI_AI智能质检_面试准备总手册.md`
- Modify: `PIS-IN_AOI_AI智能质检_面试专业题库与落地复盘.md`
- Modify: `PIS-IN_AOI_AI智能质检_面试10问10答_统一版.md`
- Create: `docs/quality/metric-dictionary.md`
- Test: `docs/tests/metric_consistency.ps1`

**Interfaces:**
- Produces one canonical metric dictionary for all downstream documents

- [ ] **Step 1: Write the failing consistency script**

```powershell
$files = @(
  Get-Item '技术方案设计书.md'
  Get-ChildItem -File 'PIS-IN_AOI_AI智能质检_*.md'
  Get-Item 'tools/docs/build_v35_pdf.py' -ErrorAction SilentlyContinue
) | Where-Object { $_ }
$old = $files | Select-String -Pattern '42%\s*(->|→|降至)\s*9%'
if ($old) { throw "Old false-positive claim remains: $($old.Path -join ', ')" }
$target = Get-Content -Raw -Encoding UTF8 'docs/quality/metric-dictionary.md'
foreach ($value in @('12%', '≤6%', '≤3%', '≤1.5%', '≤0.5%')) {
  if (-not $target.Contains($value)) { throw "Missing metric value $value" }
}
```

- [ ] **Step 2: Run the audit**

Run: `powershell -File docs/tests/metric_consistency.ps1`
Expected: FAIL because old wording exists and the dictionary is absent.

- [ ] **Step 3: Write the metric dictionary and update all claims**

The dictionary must define formula, denominator, baseline, stage targets, full-inspection false-positive rate, auto-PASS escape rate, defect recall, REVIEW ratio, P95, throughput, backlog, and silent mismatch.

- [ ] **Step 4: Run the audit and commit**

Run: `powershell -File docs/tests/metric_consistency.ps1`
Expected: PASS.

```bash
git add 技术方案设计书.md PIS-IN_AOI_AI智能质检_*.md docs/quality docs/tests
git commit -m "docs: unify AOI quality metric definitions"
```

### Task 2: Project Master Document

**Files:**
- Create: `PIS-IN_AOI_AI智能质检项目总说明书_V3.5.md`
- Test: `docs/tests/master_document_sections.ps1`

**Interfaces:**
- Produces internal source for project introduction, modules, workflows, data flow, AI nodes, architecture, stack, team, period, budget, compute, deployment, contributions, outcomes, and advantages

- [ ] **Step 1: Write the required-section audit**

```powershell
$text = Get-Content -Raw -Encoding UTF8 'PIS-IN_AOI_AI智能质检项目总说明书_V3.5.md'
foreach ($heading in @('项目背景','核心功能模块','主要业务流程','数据流向','关键 AI 技术','技术架构','技术栈','团队与周期','预算','算力','部署','个人工作成果','个人优势')) {
  if (-not $text.Contains($heading)) { throw "Missing section $heading" }
}
```

- [ ] **Step 2: Run the audit**

Run: `powershell -File docs/tests/master_document_sections.ps1`
Expected: FAIL because the master document is absent.

- [ ] **Step 3: Write the document with Mermaid diagrams**

Include system context, business process, data flow, deployment topology, team-stage matrix, internal budget table, compute estimate, three Agent boundaries, RAG knowledge scope, frontend pages, personal contribution-to-advantage mapping, and fact/estimate/target labels.

- [ ] **Step 4: Run the audit and commit**

Run: `powershell -File docs/tests/master_document_sections.ps1`
Expected: PASS.

```bash
git add PIS-IN_AOI_AI智能质检项目总说明书_V3.5.md docs/tests/master_document_sections.ps1
git commit -m "docs: add AOI project master document"
```

### Task 3: Enhanced Technical Solution and Deployment Manual

**Files:**
- Create: `PIS-IN_AOI_AI智能质检技术方案_V3.5_落地与展示增强版.md`
- Create: `PIS-IN_AOI_AI智能质检部署运维手册_V3.5.md`
- Modify: `PIS-IN_AOI_AI智能质检系统技术方案_V3.5_数据关联一致性保障专章.md`
- Test: `docs/tests/deployment_command_check.ps1`

**Interfaces:**
- Technical solution references real application routes, Docker services, database migrations, Agent endpoints, and failure modes
- Deployment manual provides exact demo and GPU-profile commands

- [ ] **Step 1: Write command-existence checks**

```powershell
$manual = Get-Content -Raw -Encoding UTF8 'PIS-IN_AOI_AI智能质检部署运维手册_V3.5.md'
foreach ($command in @('docker compose','alembic upgrade head','npm run typecheck','python -m pytest','npx playwright test')) {
  if (-not $manual.Contains($command)) { throw "Missing command $command" }
}
```

- [ ] **Step 2: Run the check**

Run: `powershell -File docs/tests/deployment_command_check.ps1`
Expected: FAIL before documents exist.

- [ ] **Step 3: Write both documents and align the V3.5 supplement**

Document GPU-free demo, local-LLM profile, TensorRT profile, environment variables, database migration, startup, health checks, backup, recovery, rollback, Agent failure, simulator control, monitoring, and production restrictions.

- [ ] **Step 4: Run checks and commit**

Run: `powershell -File docs/tests/deployment_command_check.ps1`
Expected: PASS.

```bash
git add PIS-IN_AOI_AI智能质检技术方案_V3.5_落地与展示增强版.md PIS-IN_AOI_AI智能质检部署运维手册_V3.5.md PIS-IN_AOI_AI智能质检系统技术方案_V3.5_数据关联一致性保障专章.md docs/tests
git commit -m "docs: add enhanced solution and deployment manual"
```

### Task 4: Interview Evidence and Personal Advantage Refresh

**Files:**
- Modify: `PIS-IN_AOI_AI智能质检_简历项目描述与面试讲解.md`
- Modify: `PIS-IN_AOI_AI智能质检_面试准备总手册.md`
- Modify: `PIS-IN_AOI_AI智能质检_面试专业题库与落地复盘.md`
- Modify: `PIS-IN_AOI_AI智能质检_面试10问10答_统一版.md`
- Create: `PIS-IN_AOI_AI智能质检_个人成果证据矩阵.md`

**Interfaces:**
- Produces consistent answers for eight-person staged team, five-month period, internal budget, estimated compute, three Agents, RAG, deployment, quality targets, and personal ownership

- [ ] **Step 1: Create a claim-evidence matrix**

Each row must contain `面试主张`, `个人动作`, `团队协作`, `证据文件/页面`, `事实类型`, `禁止过度承诺`. Include at least requirements, Source Key, YOLO/TensorRT, Agent/RAG, frontend, deployment, quality metrics, and cross-team delivery.

- [ ] **Step 2: Update resume and interview answers**

Use first-person action verbs and explicitly separate personal deliverables from team results. Replace generic team language with the confirmed eight roles and staged participation.

- [ ] **Step 3: Run metric consistency and manual review**

Run: `powershell -File docs/tests/metric_consistency.ps1`
Expected: PASS; no customer-facing paragraph exposes budget.

- [ ] **Step 4: Commit interview materials**

```bash
git add PIS-IN_AOI_AI智能质检_*.md
git commit -m "docs: strengthen AOI interview evidence"
```

### Task 5: Capture Verified Frontend Screenshots

**Files:**
- Create: `apps/web/tests/e2e/capture-doc-screens.spec.ts`
- Create: `docs/assets/ui/`
- Modify: `.gitignore`

**Interfaces:**
- Produces screenshots: `dashboard.png`, `inspection.png`, `tray-map.png`, `review.png`, `defect-reports.png`, `alerts-report.png`, `model-governance.png`, `project-profile.png`

- [ ] **Step 1: Write deterministic screenshot setup**

```ts
test.beforeEach(async ({request}) => {
  await request.post("http://localhost:8080/api/v1/demo/reset", {data: {seed: 202408}})
})
```

- [ ] **Step 2: Capture desktop and mobile views**

Run: `cd apps/web && npx playwright test tests/e2e/capture-doc-screens.spec.ts`
Expected: eight named screenshots with stable seeded data.

- [ ] **Step 3: Inspect every screenshot**

Verify no overlap, clipping, blank charts, broken icons, budget text, or dynamic layout shift at desktop and mobile widths.

- [ ] **Step 4: Commit selected screenshots**

```bash
git add apps/web/tests/e2e/capture-doc-screens.spec.ts docs/assets/ui .gitignore
git commit -m "docs: capture verified AOI application screens"
```

### Task 6: Regenerate and Visually Verify V3.5 PDFs

**Files:**
- Create: `tools/docs/build_v35_pdf.py`
- Produce: `PIS-IN_AOI_AI智能质检技术方案_V3.5_客户版.pdf`
- Produce: `PIS-IN_AOI_AI智能质检项目展示_V3.5.pdf`
- Create: `tmp/pdfs/qa-v35-final/`

**Interfaces:**
- Technical PDF includes architecture, data association, Agent/RAG boundaries, deployment modes, quality targets, and selected UI evidence
- Showcase PDF includes project story, runnable system, team/period, contributions, outcomes, and advantages; excludes budget

- [ ] **Step 1: Update PDF source content**

Replace old metric cards, add a staged-target visual, add runnable-system and Agent/RAG pages, and insert verified frontend screenshots. Keep customer-facing pages free of budget details.

- [ ] **Step 2: Generate both PDFs**

Run: `python tools/docs/build_v35_pdf.py`
Expected: both root PDFs are recreated and non-empty.

- [ ] **Step 3: Render every page to PNG**

Run:

```bash
pdftoppm -png -r 150 PIS-IN_AOI_AI智能质检技术方案_V3.5_客户版.pdf tmp/pdfs/qa-v35-final/tech
pdftoppm -png -r 150 PIS-IN_AOI_AI智能质检项目展示_V3.5.pdf tmp/pdfs/qa-v35-final/show
```

Expected: PNG count equals the sum of PDF page counts.

- [ ] **Step 4: Inspect every rendered page and run text checks**

Use pypdf to assert both PDFs contain `V3.5`, `12%`, `6%`, `3%`, `1.5%`, and no `V3.0` or old `42% -> 9%` wording. Inspect all PNG pages at full size for clipping, overlap, missing glyphs, broken tables, stale screenshots, and inconsistent page numbering.

- [ ] **Step 5: Commit final documents and PDFs**

```bash
git add tools/docs/build_v35_pdf.py PIS-IN_AOI_AI智能质检技术方案_V3.5_客户版.pdf PIS-IN_AOI_AI智能质检项目展示_V3.5.pdf
git commit -m "docs: publish enhanced V3.5 AOI package"
```
