import pstats
p = pstats.Stats('demo_profile.txt')
p.strip_dirs().sort_stats('cumtime').print_stats()