# Skill Authoring Guide — Connector Requirements

This guide covers how to declare connector and tool requirements in your
`SKILL.md` so IDP Kit can:

1. Show users a pre-install compatibility checklist.
2. Tell them exactly which integrations to connect.
3. Inject the right tools into IDA's prompt at runtime.

## SKILL.md frontmatter

Connector requirements live in the YAML frontmatter at the top of your
`SKILL.md`. Three forms are accepted (you can mix them):

```yaml
---
name: daily-slack-digest
description: Posts a daily summary of new documents to a Slack channel.
requires:
  connectors: [slack]
  tools: [slack_send_message, search_document]
allowed-tools: [slack_send_message]   # alternate spelling, additive
connectors: [notion]                  # top-level shorthand, additive
---
```

### Field reference

| Field                | Type        | Meaning                                                                                        |
|----------------------|-------------|------------------------------------------------------------------------------------------------|
| `requires.connectors`| list of str | Connector ids the skill needs (e.g. `slack`, `notion`, `github`).                              |
| `requires.tools`     | list of str | Specific tool names the skill calls. Connector ids are inferred from tool prefixes.            |
| `allowed-tools`      | list/str    | Claude-Code-style allow-list. Treated as additive to `requires.tools`.                         |
| `connectors`         | list of str | Top-level shorthand for `requires.connectors`.                                                 |

When you list a tool name like `slack_send_message`, IDP Kit infers the
`slack` connector automatically — you don't need to list it twice.

## Available connectors

| Id          | Display name      | Auth type    | Example tools                             |
|-------------|-------------------|--------------|-------------------------------------------|
| `slack`     | Slack             | bot token    | `slack_send_message`, `slack_list_channels` |
| `notion`    | Notion            | int. token   | `notion_search_pages`, `notion_create_page` |
| `github`    | GitHub            | PAT          | `github_list_repos`, `github_create_issue`, `github_search_code` |
| `linear`    | Linear            | API key      | `linear_list_issues`, `linear_create_issue` |
| `hubspot`   | HubSpot           | private app  | `hubspot_search_contacts`, `hubspot_create_contact` |
| `dropbox`   | Dropbox           | access token | `dropbox_list_files`, `dropbox_create_shared_link` |
| `s3`        | AWS S3            | composite    | `s3_list_objects`, `s3_presigned_url`     |
| `google`    | Google Workspace  | OAuth2       | `google_drive_search`, `google_gmail_send` |
| `jira`      | Jira              | composite    | `jira_search_issues`, `jira_create_issue` |

> The `GET /api/connectors` endpoint returns the live list with full tool
> schemas — use it to discover newly-added connectors.

## How requirements are used at runtime

1. **At import.** The skill importer parses your frontmatter and stores
   the normalised `{connectors: [...], tools: [...]}` payload on the
   `Skill` row. The import preview UI cross-checks these against the
   user's active connections and shows a green/red checklist.
2. **At chat time.** When IDA loads, only tools from the user's
   *connected* integrations are exposed to the LLM. The system prompt
   also lists which connectors are *not* connected so the model can
   tell the user how to set them up rather than fabricate output.
3. **Per call.** Credentials are decrypted just-in-time. They never
   appear in logs or in the LLM's message history. If the connector
   returns an auth failure, the connection is automatically marked
   `disconnected` and the user is prompted to reconnect.

## Best practices

- **Declare every connector your skill calls.** This lets IDP Kit warn
  users *before* they install the skill.
- **Prefer specific tool names** in `requires.tools` over connector-id
  globs — this lets the prompt advertise capabilities accurately.
- **Don't put credentials in your SKILL.md.** Skills are user-shared
  documents. All real auth happens via the Connections page.
- **Fail loud on missing connectors.** Inside your skill instructions,
  tell IDA to stop and explain which integration the user must connect
  if a required tool is missing.

## Example: a skill that posts a digest to Slack

```yaml
---
name: daily-digest
description: Posts a Slack message summarising today's new uploads.
requires:
  connectors: [slack]
  tools: [slack_send_message, list_documents]
---

# Daily Digest

1. Call `list_documents` to find documents created today.
2. Summarise each into one sentence.
3. Post the summary to the channel the user requests via
   `slack_send_message(channel=..., text=...)`.

If `slack_send_message` is not available, stop and tell the user:
"Connect Slack at /connections first, then re-run this skill."
```
