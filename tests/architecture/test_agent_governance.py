from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / ".codex" / "agents"


def test_governance_has_six_non_overlapping_role_templates():
    expected = {
        "research_manager.toml": "research_manager",
        "scientific_analyst.toml": "scientific_analyst",
        "data_bioinformatics_steward.toml": "data_bioinformatics_steward",
        "model_engineer.toml": "model_engineer",
        "evaluation_statistics_analyst.toml": "evaluation_statistics_analyst",
        "reviewer.toml": "independent_reviewer",
    }
    assert not (AGENT_DIR / "engineer.toml").exists()
    assert {path.name for path in AGENT_DIR.glob("*.toml")} == set(expected)
    templates = {path.name: tomllib.loads(path.read_text()) for path in AGENT_DIR.glob("*.toml")}
    assert {name: template["name"] for name, template in templates.items()} == expected
    assert templates["reviewer.toml"]["sandbox_mode"] == "read-only"


def test_adaptive_parallel_budget_and_gates_are_governed():
    root_rules = (ROOT / "AGENTS.md").read_text()
    governance = (ROOT / "docs" / "AGENT_GOVERNANCE.md").read_text()
    assert "DEFAULT_AGENT_BUDGET" not in root_rules
    assert "ADAPTIVE_AGENT_BUDGET" in root_rules
    assert "独立任务必须优先" in root_rules
    assert "并行；只串行化真实依赖" in root_rules
    assert "Minimal Context Principle" in governance
    assert "structured evidence package" in governance
    assert "PRIMARY_SOURCE_VERIFICATION" in governance
    assert "Role availability != Role activation" in governance
    assert "IDEA -> Manager -> EXP-ID -> APPROVE -> parallel evidence work -> Reviewer" in governance


def test_exp_scoped_agent_lifecycle_is_governed():
    root_rules = (ROOT / "AGENTS.md").read_text()
    governance = (ROOT / "docs" / "AGENT_GOVERNANCE.md").read_text()
    templates = "\n".join(path.read_text() for path in AGENT_DIR.glob("*.toml"))

    assert "EXP_SCOPED_AGENT_LIFECYCLE" in root_rules
    assert "AGENT_EXECUTION_MANIFEST" in root_rules
    assert "CROSS_EXP_AGENT_REUSE_JUSTIFICATION" in root_rules
    assert "Manager execution" in root_rules
    assert "dry-run Agent" in root_rules
    assert "fresh context" in governance
    assert "actual child agent spawned" in governance
    assert "EXP-ID + Role" in governance
    assert "实验关闭后" in root_rules
    assert "禁止跨 EXP 复用" in templates


def test_large_data_runs_require_a_dry_run_and_reuse_rule():
    governance = (ROOT / "docs" / "AGENT_GOVERNANCE.md").read_text()
    assert "SMALL-SCALE DRY RUN" in governance
    assert "restart/recovery" in governance
    assert "重新扫描完整源矩阵" in governance


def test_github_audit_checkpoints_are_governed():
    root_rules = (ROOT / "AGENTS.md").read_text()
    governance = (ROOT / "docs" / "AGENT_GOVERNANCE.md").read_text()
    for text in (root_rules, governance):
        assert "DESIGN READY" in text
        assert "RESULT REVIEWED" in text
        assert "GitHub" in text
        assert "checksum" in text
    assert "等待 `APPROVE EXP-###`" in governance
    assert "等待用户 `ACCEPT / REJECT`" in governance


def test_coarse_to_fine_policy_distinguishes_fast_mvp_and_rigorous_loops():
    root_rules = (ROOT / "AGENTS.md").read_text()
    governance = (ROOT / "docs" / "AGENT_GOVERNANCE.md").read_text()
    mvp = (ROOT / "mvp" / "records" / "MVP-001.md").read_text()
    for text in (root_rules, governance):
        assert "COARSE_TO_FINE_ML_RESEARCH_POLICY" in text
        assert "FAST LOOP" in text
        assert "MVP LOOP" in text
        assert "RIGOROUS LOOP" in text
    assert "不是 Formal Research EXP" in mvp
    assert "result_reviewed" in mvp
    assert "DEFERRED_PDO_LEG" in mvp


def test_every_experiment_record_has_execution_manifest():
    records = sorted((ROOT / "experiments" / "records").glob("EXP-*.md"))
    assert records
    for record in records:
        text = record.read_text()
        assert "AGENT_EXECUTION_MANIFEST" in text, record
        assert "spawned_for_exp" in text, record
        assert "fresh_context" in text, record
        assert "worktree_isolation" in text, record
