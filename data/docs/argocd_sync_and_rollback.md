# ArgoCD Sync and Rollback

## Syncing an application
An ArgoCD Application continuously compares the live cluster state against the desired
state declared in Git. When they differ the app is reported OutOfSync. A sync applies the
Git-declared manifests to the cluster. With `syncPolicy.automated` enabled, ArgoCD syncs
automatically; `selfHeal: true` reverts manual cluster changes back to Git, and
`prune: true` deletes resources removed from Git.

## Rolling back an application
Each successful sync is recorded in the app's history. To roll back, select a previous
history revision and sync to it, or run `argocd app rollback <app> <revision>`. Rollback
re-applies the manifests from that revision. If automated sync with self-heal is on,
disable automation first, otherwise ArgoCD will immediately re-sync back to the latest
Git commit and undo the rollback.

## Health and sync status
ArgoCD reports two independent signals: sync status (does the cluster match Git) and health
(are the resources actually healthy, e.g. a Deployment's pods available). An app can be
Synced but Degraded, or Healthy but OutOfSync.
