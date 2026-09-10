# Kubernetes Deployments and Rollouts

## Rolling updates
A Deployment updates pods with a rolling strategy by default, governed by `maxSurge` and
`maxUnavailable`. New ReplicaSets scale up while old ones scale down, keeping the app
available throughout.

## Rollback
`kubectl rollout undo deployment/<name>` reverts to the previous ReplicaSet.
`kubectl rollout history deployment/<name>` lists revisions; add `--revision=N` for detail.

## Readiness and liveness probes
A readiness probe gates whether a pod receives traffic; a failing readiness probe removes
the pod from Service endpoints without restarting it. A liveness probe restarts a pod that
is alive but stuck. Separating the two avoids restart loops during slow warmups.
