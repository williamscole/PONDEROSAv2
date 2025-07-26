#!/bin/bash

# Simple Class and Function Analyzer
# Compatible with older bash versions

TARGET_DIR=${1:-.}

echo "🏗️  CLASS AND FUNCTION ANALYSIS"
echo "================================"
echo

# Function to analyze a single file
analyze_file() {
    local file="$1"
    echo "📄 File: $(basename "$file")"
    echo "   Path: $file"
    
    # Find classes
    echo "   Classes:"
    grep -n "^class " "$file" 2>/dev/null | while IFS=: read -r line_num content; do
        class_name=$(echo "$content" | awk '{print $2}' | sed 's/[(:].*//')
        if echo "$content" | grep -q "("; then
            inheritance=$(echo "$content" | sed 's/.*(//' | sed 's/).*//')
            echo "      🔸 Line $line_num: $class_name (inherits from: $inheritance)"
        else
            echo "      🔸 Line $line_num: $class_name"
        fi
    done
    
    # Find top-level functions
    echo "   Functions:"
    grep -n "^def " "$file" 2>/dev/null | while IFS=: read -r line_num content; do
        func_name=$(echo "$content" | awk '{print $2}' | sed 's/(.*//')
        echo "      🔧 Line $line_num: $func_name()"
    done
    
    # Find methods (indented functions)
    echo "   Methods:"
    grep -n "^    def " "$file" 2>/dev/null | while IFS=: read -r line_num content; do
        method_name=$(echo "$content" | awk '{print $2}' | sed 's/(.*//')
        echo "      ⚙️  Line $line_num: $method_name()"
    done | head -10  # Limit to first 10 methods
    
    echo
}

# Analyze each Python file
echo "Scanning for Python files in: $TARGET_DIR"
echo

for file in $(find "$TARGET_DIR" -name "*.py" | sort); do
    if [ -f "$file" ]; then
        analyze_file "$file"
    fi
done

echo
echo "🔗 IMPORT ANALYSIS"
echo "=================="

for file in $(find "$TARGET_DIR" -name "*.py" | sort); do
    echo "📄 $(basename "$file"):"
    
    # Find imports
    grep -n "^import \|^from .* import" "$file" 2>/dev/null | head -5 | while IFS=: read -r line_num content; do
        echo "   📦 Line $line_num: $content"
    done
    echo
done

echo "📊 QUICK STATS"
echo "=============="
total_files=$(find "$TARGET_DIR" -name "*.py" | wc -l)
total_classes=$(find "$TARGET_DIR" -name "*.py" -exec grep -c "^class " {} \; | awk '{sum+=$1} END {print sum}')
total_functions=$(find "$TARGET_DIR" -name "*.py" -exec grep -c "^def " {} \; | awk '{sum+=$1} END {print sum}')

echo "Python files: $total_files"
echo "Total classes: $total_classes"
echo "Total top-level functions: $total_functions"
