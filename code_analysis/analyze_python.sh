#!/bin/bash

# Script to analyze Python codebase structure
# Usage: ./analyze_python.sh [directory]

TARGET_DIR=${1:-.}

echo "=== Python Code Analysis for: $TARGET_DIR ==="
echo

# 1. Find all Python files
echo "📁 Python files found:"
find "$TARGET_DIR" -name "*.py" | sort
echo

# 2. Extract class names from each Python file
echo "🏗️  Class definitions by file:"
echo "================================"
for file in $(find "$TARGET_DIR" -name "*.py" | sort); do
    echo "📄 $file:"
    # Look for class definitions (handles inheritance too)
    grep -n "^class " "$file" 2>/dev/null | sed 's/^/  /' || echo "  No classes found"
    echo
done

# 3. Extract function definitions (top-level only)
echo "🔧 Top-level function definitions:"
echo "=================================="
for file in $(find "$TARGET_DIR" -name "*.py" | sort); do
    echo "📄 $file:"
    # Look for function definitions at start of line (top-level functions)
    grep -n "^def " "$file" 2>/dev/null | sed 's/^/  /' || echo "  No top-level functions found"
    echo
done

# 4. Find imports to understand dependencies
echo "📦 Import statements:"
echo "===================="
for file in $(find "$TARGET_DIR" -name "*.py" | sort); do
    echo "📄 $file:"
    # Find import statements
    grep -E "^(import|from)" "$file" 2>/dev/null | sed 's/^/  /' || echo "  No imports found"
    echo
done

# 5. Look for main execution blocks
echo "🚀 Main execution blocks:"
echo "========================"
for file in $(find "$TARGET_DIR" -name "*.py" | sort); do
    if grep -q "if __name__ == ['\"]__main__['\"]" "$file" 2>/dev/null; then
        echo "  ✅ $file has main execution block"
    fi
done
echo

# 6. Count lines of code
echo "📊 Code statistics:"
echo "=================="
total_lines=0
total_files=0
for file in $(find "$TARGET_DIR" -name "*.py"); do
    lines=$(wc -l < "$file")
    total_lines=$((total_lines + lines))
    total_files=$((total_files + 1))
    printf "  %-40s %6d lines\n" "$(basename "$file")" "$lines"
done
echo "  ----------------------------------------"
printf "  %-40s %6d lines\n" "TOTAL ($total_files files)" "$total_lines"
echo
