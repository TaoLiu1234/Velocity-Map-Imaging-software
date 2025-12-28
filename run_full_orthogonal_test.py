"""Run full orthogonal test with reduced events for faster execution."""
import sys
import test_orthogonal_performance as t

# Reduce event count for faster testing (100k instead of 1M)
# This makes the full test suite run in ~30 minutes instead of hours
t.N_EVENTS = int(1e5)

print("=" * 80)
print("QUICK ORTHOGONAL TEST (100k events per test)")
print("=" * 80)

# Run main
t.main()
