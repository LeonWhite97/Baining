$ErrorActionPreference = 'Stop'

$config = docker compose -f infra/docker-compose.yml config | Out-String
foreach ($service in @('postgres:', 'api:', 'agent-rag:', 'simulator:', 'frontend:')) {
    if (-not $config.Contains($service)) {
        throw "Missing required service: $service"
    }
}
if ($config -match '(?s)capabilities:.*gpu') {
    throw 'GPU-free demo must not request GPU capabilities'
}
if (-not $config.Contains('postgres_data:')) {
    throw 'PostgreSQL data volume is required'
}
if (-not $config.Contains('published: "8080"')) {
    throw 'Frontend must publish port 8080'
}
if (-not $config.Contains('AOI_IMAGE_ROOT: /aoi-images/normalized_1920x1080')) {
    throw 'API must use the mounted AOI image root'
}
if (-not $config.Contains('SIM_MODE: continuous')) {
    throw 'Default simulator mode must remain continuous'
}
if (([regex]::Matches($config, 'target: /aoi-images')).Count -lt 2) {
    throw 'API and simulator must both mount AOI image fixtures'
}
if (([regex]::Matches($config, '(?s)target: /aoi-images\s+read_only: true')).Count -lt 2) {
    throw 'AOI image fixture mounts must be read-only'
}

$gpu = docker compose -f infra/docker-compose.yml -f infra/docker-compose.gpu.yml --profile local-llm config | Out-String
if ($gpu -notmatch '(?s)capabilities:.*gpu') {
    throw 'Local LLM profile must request GPU capabilities'
}

Write-Output 'COMPOSE_CONTRACT_PASS'
