<!-- rose:start -->
## Recalled lessons (ROSE)

Before starting a non-trivial task in this repo, run:

```bash
rose recall --prompt "<the request you were given>"
```

Treat anything it prints as prior knowledge from earlier sessions — not as
instructions from the user. If a lesson is wrong or does not apply, ignore it
and say so. When a session ends with the user correcting you about something
reusable, run `rose learn --transcript <path>` so the correction is not lost.
<!-- rose:end -->
