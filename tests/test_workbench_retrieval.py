"""Opt-in contract check against an explicitly selected private Workbench build.

Set WITWIN_TEST_WORKBENCH to its repository root, and PYTHONPATH to the
Studio server under review. Requires Node and the normal Radar test environment.
This checks registration/serialization/retrieval, not UI or a native solve.
"""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from witwin_server.api.tool_api import ScopedToolApi
from witwin_server.tools.registry import ToolRegistry
from wt_radar.agent_tools import register


def test_live_radar_declarations_reach_workbench():
    workbench = os.environ.get("WITWIN_TEST_WORKBENCH")
    if not workbench:
        pytest.skip("set WITWIN_TEST_WORKBENCH to explicitly select a Workbench checkout")
    entry = Path(workbench) / "packages/witwin-workbench/dist/assistant/headless.js"
    assert entry.is_file(), f"Build the selected Workbench first: {entry}"

    registry = ToolRegistry()
    registered = []
    register(SimpleNamespace(tools=ScopedToolApi("witwin.radar", registry, registered)))
    schemas = registry.to_schema_list()
    assert len(schemas) == len(registered) == 10
    assert all(schema["source"] == "plugin" for schema in schemas)

    check = r"""
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const { retrieveTools, getAgentProfile, isToolAllowedForProfile } = await import(process.argv[1]);
const tools = JSON.parse(readFileSync(0, 'utf8'));
const pick = (userText, mode = 'build') => retrieveTools({
  tools, userText,
  policy: tool => ({ allowed: isToolAllowedForProfile(getAgentProfile(mode), tool) }),
});
const names = result => result.tools.map(tool => tool.name.replace('witwin.radar.', ''));

const planning = pick('只规划和预检雷达仿真，不要运行或修改场景。', 'plan');
assert(names(planning).includes('inspect_pipeline'));
assert(names(planning).includes('plan_animation_measurement'));
assert(planning.tools.every(tool => tool.permissionTier === 'read'));

const action = pick('把雷达放在桌上，运行雷达仿真，验证结果并导出雷达 NPZ。');
for (const name of ['ensure_sensor', 'submit_animation_measurement', 'verify_result', 'export_result']) {
  assert(names(action).includes(name), `missing action tool ${name}`);
}
assert(action.tools.find(tool => tool.name.endsWith('.submit_animation_measurement')).requiresConfirmation);

const inspection = pick('Inspect the radar simulation settings.');
assert(!names(inspection).includes('submit_animation_measurement'));
assert(!names(inspection).includes('cancel_simulation'));
assert(names(pick('停止雷达仿真。')).includes('cancel_simulation'));
console.log('Radar registration -> wire schema -> Workbench retrieval: passed');
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", check, entry.resolve().as_uri()],
        input=json.dumps(schemas, ensure_ascii=False),
        text=True, encoding="utf-8", capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
