| genre | strategy | model | prompt | policy | detector | n | before | after | evaded_of_detected | rewrite_accepted | mean_similarity_kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| story | full | qwen/qwen3.8-27b | v1 | kept | ours | 34 | 97% (85%-99%) | 6% (2%-19%) | 94% | 94% | 0.87 |
| story | full | qwen/qwen3.8-27b | v1 | kept | desklib | 34 | 100% (90%-100%) | 68% (51%-81%) | 32% | 94% | 0.87 |
| story | full | qwen/qwen3.8-27b | v1 | unguarded | ours | 34 | 97% (85%-99%) | 0% (0%-10%) | 100% | 94% | 0.87 |
| story | full | qwen/qwen3.8-27b | v1 | unguarded | desklib | 34 | 100% (90%-100%) | 47% (31%-63%) | 53% | 94% | 0.87 |
