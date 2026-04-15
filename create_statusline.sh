cat > ~/.claude/statusline.sh <<'EOF'
#!/usr/bin/env bash

input="$(cat)"

model="$(printf '%s' "$input" | jq -r '.model.display_name // "Claude"')"
dir="$(printf '%s' "$input" | jq -r '.workspace.current_dir // .cwd // "~"')"
used="$(printf '%s' "$input" | jq -r '(.context_window.used_percentage // 0) | floor')"
remaining="$(printf '%s' "$input" | jq -r '(.context_window.remaining_percentage // (100 - (.context_window.used_percentage // 0))) | floor')"
total="$(printf '%s' "$input" | jq -r '.context_window.context_window_size // 0')"

dir_name="${dir##*/}"
[ -z "$dir_name" ] && dir_name="$dir"

printf "[%s] %s | 上下文: %s%% / %s | 剩余: %s%%\n" \
  "$model" \
  "$dir_name" \
  "$used" \
  "$total" \
  "$remaining"
EOF

chmod +x ~/.claude/statusline.sh