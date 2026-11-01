param(
    [string]$Project = 'C:\WiTwin\projects\radar-ui-review',
    [int]$Port = 8011
)
$ErrorActionPreference = 'Stop'
$reviewPython = 'C:\Users\WQZ\AppData\Local\anaconda3\envs\witwin2\python.exe'
$reviewPlugin = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = 'C:\WiTwin\repos\witwin-studio\server;C:\WiTwin\plugins'
Write-Host "With the Studio frontend running, open http://localhost:3000/?server=127.0.0.1&port=$Port"
& $reviewPython -m witwin_server.server_entry serve --project $Project --plugin-path $reviewPlugin --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
