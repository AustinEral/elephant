# Security Policy

## Supported Use

The safest operating posture for this repository is:

- local development
- internal evaluation
- self-hosted deployments behind your own network and auth controls

Elephant does **not** ship with built-in API authentication, and the server uses permissive CORS by default. Do not expose it directly to the public internet without adding your own gateway and access controls.

## Reporting a Vulnerability

Please do **not** open a public GitHub issue with exploit details.

Use a private reporting path when one is available, such as a maintainer contact or a private GitHub security advisory workflow.

If you cannot report privately, open a minimal public issue without exploit details and clearly mark it as a request for a private follow-up.

Include:

- affected version or commit
- deployment mode
- steps to reproduce
- impact
- whether the issue requires valid credentials or local access

## Scope

Security reports are especially useful for:

- authentication or authorization bypass
- tenant or bank isolation problems
- remote code execution
- data exposure across banks
- unsafe default deployment behavior
- injection vulnerabilities
- dependency or supply-chain issues with practical impact

## Response Expectations

Security reports are handled without an SLA.

The goal is to:

- confirm receipt
- reproduce and assess impact
- fix or mitigate
- document the change clearly

## Operational Guidance

If you run Elephant, assume responsibility for standard production controls:

- reverse proxy / TLS
- authentication and authorization at the edge
- private networking
- database backup and restore
- secrets management
- monitoring and logging

See:

- [docs/deployment.md](docs/deployment.md)
