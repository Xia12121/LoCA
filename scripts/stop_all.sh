#!/usr/bin/env bash
# Kill all LoCA experiment processes cleanly. Invoke as `bash scripts/stop_all.sh`
# so pkill matches the experiment scripts, NOT this stopper's own command line.
for pat in pipeline_gpu pipeline_gpu2 pipeline_fixes pipeline_cpu run_all_gpu run_all_cpu phase1_matrix phase2_matrix phase2_ceiling phase2_efficiency phase3_science c2_convergence "scripts/phase1.py" "scripts/sweep.py"; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 2
echo "remaining experiment procs:"
ps -eo pid,cmd | grep -iE "phase1|phase2|phase3|pipeline_|run_all|sweep.py" | grep -v "grep\|stop_all" | head
