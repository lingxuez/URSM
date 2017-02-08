import pstats
p = pstats.Stats('demo_profile.txt')
p.strip_dirs().sort_stats('cumulative').print_stats()