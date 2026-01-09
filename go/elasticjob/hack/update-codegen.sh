#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

MODULE="github.com/intelligent-machine-learning/dlrover/go/elasticjob"
CODEGEN_PKG=/Users/xlzheng/go/pkg/mod/k8s.io/code-generator@v0.24.2
# 你的 API 定义路径
INPUT_PACKAGE="elastic.iml.github.io/v1alpha1:github.com/intelligent-machine-learning/dlrover/go/elasticjob/api/v1alpha1"

OUTPUT_PKG="${MODULE}/client"

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
BOILERPLATE="${ROOT_DIR}/hack/boilerplate.go.txt"

CODEGEN_TMP=$(mktemp -d)

export GOPATH="$CODEGEN_TMP"
# 1. 创建目标 package 的完整父路径
# 注意：这里只建目录，不建立项目根目录的软链接，防止路径重叠
BASE_DIR="$CODEGEN_TMP/src/github.com/intelligent-machine-learning/dlrover/go/elasticjob"
mkdir -p "$BASE_DIR/pkg/apis/elastic"
mkdir -p "$BASE_DIR/api"

# 2. 建立 API 目录的软链接 (为了 deepcopy-gen)
ln -s "$ROOT_DIR/api/v1alpha1" "$BASE_DIR/api/v1alpha1"

# 3. 建立伪装路径的软链接 (为了 client-gen)
# 这一步是让工具认为代码在 pkg/apis/elastic/v1alpha1
ln -s "$ROOT_DIR/api/v1alpha1" "$BASE_DIR/pkg/apis/elastic/v1alpha1"

echo "清理旧文件..."
rm -rf "${ROOT_DIR}/client"

echo "生成 deepcopy 方法..."
deepcopy-gen \
  --input-dirs $MODULE/api/v1alpha1 \
  -O zz_generated.deepcopy \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src"

#cp -r "$CODEGEN_TMP/src/$MODULE/api/v1alpha1/zz_generated.deepcopy.go" "${ROOT_DIR}/api/v1alpha1/"

echo "生成 clientset..."
client-gen \
  --clientset-name versioned \
  --input-base "" \
  --input "$MODULE/pkg/apis/elastic/v1alpha1" \
  --output-package "$MODULE/client/clientset" \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src" \
  -v 5

echo "生成 lister..."
lister-gen \
  --input-dirs "$MODULE/api/v1alpha1" \
  --output-package "$MODULE/client/listers" \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src" \
  -v 5

echo "生成 informer..."
informer-gen \
  --input-dirs "$MODULE/api/v1alpha1" \
  --versioned-clientset-package "$MODULE/client/clientset/versioned" \
  --listers-package "$MODULE/client/listers" \
  --output-package "$MODULE/client/informers" \
  --go-header-file "${BOILERPLATE}" \
  --output-base "$CODEGEN_TMP/src" \
  -v 5

mv "${CODEGEN_TMP}/src/${MODULE}/client/" "${ROOT_DIR}/"

echo "✅ 客户端代码生成完成！"

