| genre | strategy | model | prompt | policy | detector | n | before | after | evaded_of_detected | rewrite_accepted | mean_similarity_kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| story | full | qwen/qwen3.8-27b | v2 | kept | ours | 26 | 100% (87%-100%) | 12% (4%-29%) | 88% | 100% | 0.90 |
| story | full | qwen/qwen3.8-27b | v2 | kept | desklib | 26 | 100% (87%-100%) | 88% (71%-96%) | 12% | 100% | 0.90 |
| story | full | qwen/qwen3.8-27b | v2 | kept | fakespot | 26 | 88% (71%-96%) | 65% (46%-81%) | 30% | 100% | 0.90 |
| story | full | qwen/qwen3.8-27b | v2 | unguarded | ours | 26 | 100% (87%-100%) | 0% (0%-13%) | 100% | 100% | 0.90 |
| story | full | qwen/qwen3.8-27b | v2 | unguarded | desklib | 26 | 100% (87%-100%) | 77% (58%-89%) | 23% | 100% | 0.90 |
| story | full | qwen/qwen3.8-27b | v2 | unguarded | fakespot | 26 | 88% (71%-96%) | 54% (35%-71%) | 43% | 100% | 0.90 |
