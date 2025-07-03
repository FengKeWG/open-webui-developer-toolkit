# Invisible Message Encoding Pipe

Persist a secret by embedding it in a hidden markdown comment:

```markdown
[my secret]: #
```

The text inside the brackets is hidden from normal Markdown rendering. Send
another message later and the pipe will reveal it.
