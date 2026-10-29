| genre | strategy | model | policy | detector | n | before | after | evaded_of_detected | rewrite_accepted | mean_similarity_kept |
|---|---|---|---|---|---|---|---|---|---|---|
| story | full | qwen/qwen3.8-27b | kept | ours | 10 | 90% (60%-98%) | 0% (0%-28%) | 100% | 100% | 0.86 |
| story | full | qwen/qwen3.8-27b | kept | desklib | 10 | 100% (72%-100%) | 60% (31%-83%) | 40% | 100% | 0.86 |
| story | full | qwen/qwen3.8-27b | unguarded | ours | 10 | 90% (60%-98%) | 0% (0%-28%) | 100% | 100% | 0.86 |
| story | full | qwen/qwen3.8-27b | unguarded | desklib | 10 | 100% (72%-100%) | 30% (11%-60%) | 70% | 100% | 0.86 |
