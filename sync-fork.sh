#!/bin/bash
# Sync fork z upstream (repo Marcina) → zawsze tylko do brancha marcin_main
set -e
git fetch upstream
git checkout marcin_main
git merge upstream/main --no-edit
git push origin marcin_main
echo "✅ Fork zsynchronizowany: upstream/main → marcin_main"
