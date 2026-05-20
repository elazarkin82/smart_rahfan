#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Initializing and updating Yocto environment for smart_rahfan ===${NC}"

# Define the repository mappings (name, url, branch)
REPOS=(
    "poky|https://git.yoctoproject.org/poky|scarthgap"
    "meta-arm|https://git.yoctoproject.org/meta-arm|scarthgap"
    "meta-openembedded|https://git.openembedded.org/meta-openembedded|scarthgap"
    "meta-rockchip|https://github.com/radxa/meta-rockchip.git|scarthgap"
)

# Ensure yocto directory exists
mkdir -p yocto

for repo in "${REPOS[@]}"; do
    IFS="|" read -r name url branch <<< "$repo"
    target_dir="yocto/$name"
    
    if [ ! -d "$target_dir/.git" ]; then
        echo -e "${YELLOW}Cloning $name ($branch) from $url...${NC}"
        git clone --branch "$branch" "$url" "$target_dir"
        echo -e "${GREEN}Successfully cloned $name.${NC}"
    else
        echo -e "${YELLOW}Updating existing repository $name ($branch)...${NC}"
        git -C "$target_dir" fetch origin
        git -C "$target_dir" checkout "$branch"
        git -C "$target_dir" pull origin "$branch"
        echo -e "${GREEN}Successfully updated $name.${NC}"
    fi
done

echo -e "${GREEN}=== Project initialization completed successfully! ===${NC}"
