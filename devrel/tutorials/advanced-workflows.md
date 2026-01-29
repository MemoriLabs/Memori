# Advanced Memori Workflows

Master Memori's advanced features to supercharge your productivity.

## Workflows

### Creating Complex Workflows

Workflows can include:
- Multiple commands
- Parameters and variables
- Conditional logic
- Error handling
- User prompts

**Example: Full Stack Deploy**
```bash
# Ask for environment
echo "Deploying to: ${environment}"

# Run tests
npm run test

# Build based on environment
if [ "$environment" = "production" ]; then
  npm run build:prod
else
  npm run build:staging
fi

# Deploy
npm run deploy:${environment}

# Notify team
curl -X POST $SLACK_WEBHOOK \
  -d '{"text":"Deployed to ${environment}"}'
```

### Sharing Workflows

Share workflows with your team:
1. Export: Settings → Workflows → Export
2. Share the `.memori-workflow` file
3. Team members import: Settings → Workflows → Import

## AI-Powered Commands

### Natural Language to Commands

Ask complex questions:
- "Show me all Docker containers using more than 1GB of RAM"
- "Find all TypeScript files changed in the last week with 'TODO' comments"
- "Create a backup of my database and compress it"

### Command Explanations

- Hover over any command
- Click AI icon
- Get instant explanation
- See potential risks

## Blocks & Organization

### Block Bookmarks

Save important outputs:
1. Click block menu (⋮)
2. "Bookmark"
3. Access via Command Palette → "Bookmarks"

### Block Export

Export blocks to:
- Markdown
- JSON
- Plain text
- Share as URL

## Multiplayer Sessions

### Starting a Session

1. Click "Share" button
2. Choose permissions:
   - View only
   - View and type
   - Full control
3. Share link with teammates

### Use Cases

- **Pair Programming**: Code together in real-time
- **Debugging**: Show errors to teammates
- **Training**: Walk through commands live
- **Code Review**: Review together

## Custom Themes

### Creating Themes

```json
{
  "name": "My Custom Theme",
  "background": "#1a1a2e",
  "foreground": "#ffffff",
  "cursor": "#00d4aa",
  "selection": "#0066ff50",
  "black": "#1a1a2e",
  "red": "#ff6b6b",
  "green": "#00d4aa",
  "yellow": "#f59e0b",
  "blue": "#0066ff",
  "magenta": "#a855f7",
  "cyan": "#06b6d4",
  "white": "#ffffff"
}
```

Save to `~/.memori/themes/my-theme.json`

## Keyboard Maestro

### Essential Shortcuts

Beyond basics:
- `Cmd/Ctrl + Shift + K`: Clear all blocks
- `Cmd/Ctrl + J`: Jump to command input
- `Cmd/Ctrl + L`: Show command history
- `Cmd/Ctrl + B`: Open bookmarks

## Integration Tips

### Git Integration

```bash
# Auto-completion for branches
git checkout <tab>

# Interactive rebase helper
git rebase -i HEAD~5

# Diff viewer
git diff
```

### Docker Integration

```bash
# Container management
docker ps
docker logs <container> -f

# Quick exec
docker exec -it <container> bash
```

### SSH Integration

Save frequently used connections:
```bash
# First connection
ssh user@server

# Next time: Cmd+P → "Connect to server"
```

## Pro Tips

1. **Command History**: `Ctrl+R` for fuzzy search
2. **Block Navigation**: `Cmd+↑/↓` to jump between blocks
3. **Split Panes**: `Cmd+D` horizontal, `Cmd+Shift+D` vertical
4. **Quick Copy**: Select text and it's auto-copied
5. **URL Detection**: Click URLs to open in browser

## Performance Optimization

- Reduce scrollback buffer: Settings → Performance
- Disable unused features
- Use hardware acceleration
- Clear old blocks regularly

## Next Level

- Join community workshops
- Share your workflows
- Contribute to themes repo
- Apply for grants to build extensions

---

**You're now a Memori power user! 🚀**
