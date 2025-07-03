# Functions Guide

This directory groups reusable **pipes** and **filters** for Open WebUI. Both are lightweight Python modules that can be copy-pasted into a WebUI instance.

- **Pipes** transform or generate chat messages. They can call external APIs
  and emit new output.
- **Filters** inspect or modify existing messages. Filters run before and after
  a pipe to change its behavior.
