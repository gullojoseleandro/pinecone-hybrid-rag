# Kubernetes rollout

Kubernetes uses readiness probes to remove an API pod from service during startup or a database outage. A rolling deployment keeps old pods available until new pods pass readiness. Resource limits prevent one pod from exhausting node memory.
