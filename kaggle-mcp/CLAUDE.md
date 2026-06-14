# Kaggle MCP Server

Path: `/home/fadhil/program/kaggle-mcp/server.py`

38 MCP tools wrapping the Kaggle CLI for Claude Code. Registered in
`~/.claude/settings.json`.

## Tool categories

| Category       | Tools                                                                                  |
|----------------|----------------------------------------------------------------------------------------|
| Competitions   | list, files, download, submit, submissions, leaderboard                               |
| Datasets       | list, files, download, create, version, status, metadata                              |
| Kernels        | list, list_files, push, pull, output, status, logs, **logs_tail**, delete             |
| Models         | list, get, instances_list, instances_versions_list, instances_versions_download        |
| Config/Auth    | config_view, config_set, config_unset, quota, print_access_token, revoke              |
| **Profiles**   | **list, add, remove, use, show**                                                      |

## Usage

Tools named `kaggle_<group>_<command>`. All optional/required args map to the CLI flags.
Output is returned as plain text, same as what `kaggle` prints.

## Multi-user Profiles

The server supports up to 4 Kaggle users via named profiles. Each profile stores
a KGAT_... token. All tools automatically use the currently active profile's token.

| Tool | Description |
|------|-------------|
| `kaggle_profiles_list()` | List all profiles, showing which is active |
| `kaggle_profile_add(name, token)` | Add a new profile (token must be `KGAT_...`) |
| `kaggle_profile_remove(name)` | Remove a profile |
| `kaggle_profile_use(name)` | Switch the active profile for all subsequent calls |
| `kaggle_profile_show()` | Show the currently active profile name |

**Workflow:**
1. Add your profiles: `kaggle_profile_add(name="user1", token="KGAT_xxxx...")`
2. Switch between them: `kaggle_profile_use(name="user2")`
3. Run any Kaggle tool — it uses the active profile's credentials

Profiles are persisted in `~/.kaggle/profiles.json`. On first run the server
auto-migrates the existing `~/.kaggle/access_token` or `KAGGLE_API_TOKEN` env
var into a profile named `"default"`.

## First-time setup

```bash
kaggle auth login    # OAuth flow — opens a browser
```
