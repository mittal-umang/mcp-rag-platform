# Helm Charts

## Charts, releases, values
A chart packages Kubernetes manifests as templates. Installing a chart creates a release;
`values.yaml` supplies the parameters templates render against, overridable at install time
with `--set` or `-f`.

## Upgrade and rollback
`helm upgrade` renders the chart with new values and applies the diff. `helm rollback
<release> <revision>` reverts to a prior release revision. `helm history <release>` lists them.

## Templating
Templates use Go templating with the Sprig function library. `{{ .Values.x }}` reads a
value; conditionals like `{{- if .Values.feature.enabled }}` render blocks only when a flag
is set, which is how optional components are gated.
