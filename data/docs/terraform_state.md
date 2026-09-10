# Terraform State

## What state is
Terraform records the real-world resources it manages in a state file, mapping
configuration to actual infrastructure IDs. Plans diff desired config against state.

## Remote state and locking
Storing state remotely (e.g. an S3 backend with a DynamoDB lock table) lets a team share
one source of truth and prevents concurrent applies from corrupting it via state locking.

## Drift
Drift is when real infrastructure diverges from state (a manual change in the console).
`terraform plan` surfaces drift as a proposed diff; `terraform apply` reconciles it back
to the declared configuration.
