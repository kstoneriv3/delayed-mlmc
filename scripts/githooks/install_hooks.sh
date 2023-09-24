#!/bin/bash

_GITROOT=$(git rev-parse --show-toplevel)
cd $_GITROOT
cp ./scripts/githooks/pre-commit .git/hooks
cp ./scripts/githooks/pre-push .git/hooks
