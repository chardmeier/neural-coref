import mention_ranking
import pydevd

pydevd.settrace('localhost', port=21000, stdoutToServer=True, stderrToServer=True)
mention_ranking.main()
