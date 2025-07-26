#!/usr/bin/env python3
"""
Advanced Python code analyzer using AST parsing
Usage: python analyze_code.py [directory]
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        self.class_methods = defaultdict(list)
        self.current_class = None
        
    def visit_ClassDef(self, node):
        class_info = {
            'name': node.name,
            'line': node.lineno,
            'bases': [self.get_name(base) for base in node.bases],
            'methods': [],
            'decorators': [self.get_name(dec) for dec in node.decorator_list]
        }
        
        self.current_class = node.name
        self.classes.append(class_info)
        
        # Visit child nodes to get methods
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_info = {
                    'name': child.name,
                    'line': child.lineno,
                    'args': [arg.arg for arg in child.args.args],
                    'decorators': [self.get_name(dec) for dec in child.decorator_list]
                }
                class_info['methods'].append(method_info)
                self.class_methods[node.name].append(child.name)
        
        self.current_class = None
        
    def visit_FunctionDef(self, node):
        if self.current_class is None:  # Top-level function
            func_info = {
                'name': node.name,
                'line': node.lineno,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [self.get_name(dec) for dec in node.decorator_list]
            }
            self.functions.append(func_info)
            
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
            
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                'type': 'from_import',
                'module': node.module,
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
    
    def get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)

def analyze_file(file_path):
    """Analyze a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        return {
            'file': str(file_path),
            'classes': analyzer.classes,
            'functions': analyzer.functions,
            'imports': analyzer.imports,
            'lines': len(content.split('\n'))
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'classes': [],
            'functions': [],
            'imports': [],
            'lines': 0
        }

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    target_path = Path(target_dir)
    
    print("🐍 PYTHON CODE ANALYSIS")
    print("=" * 50)
    
    results = []
    python_files = list(target_path.glob('**/*.py'))
    
    for py_file in sorted(python_files):
        result = analyze_file(py_file)
        results.append(result)
    
    # Summary
    total_classes = sum(len(r['classes']) for r in results)
    total_functions = sum(len(r['functions']) for r in results)
    total_lines = sum(r['lines'] for r in results)
    
    print(f"📊 SUMMARY")
    print(f"Files analyzed: {len(results)}")
    print(f"Total classes: {total_classes}")
    print(f"Total functions: {total_functions}")
    print(f"Total lines: {total_lines}")
    print()
    
    # Detailed breakdown
    for result in results:
        if result.get('error'):
            print(f"❌ {result['file']}: {result['error']}")
            continue
            
        print(f"📄 {result['file']} ({result['lines']} lines)")
        
        # Classes
        if result['classes']:
            print("   🏗️  Classes:")
            for cls in result['classes']:
                inheritance = f" → {', '.join(cls['bases'])}" if cls['bases'] else ""
                print(f"      • {cls['name']}{inheritance} (line {cls['line']})")
                if cls['methods']:
                    for method in cls['methods'][:3]:  # Show first 3 methods
                        args_str = ', '.join(method['args']) if method['args'] else ''
                        print(f"         ↳ {method['name']}({args_str})")
                    if len(cls['methods']) > 3:
                        print(f"         ↳ ... and {len(cls['methods']) - 3} more methods")
        
        # Top-level functions
        if result['functions']:
            print("   🔧 Functions:")
            for func in result['functions'][:3]:  # Show first 3
                args_str = ', '.join(func['args']) if func['args'] else ''
                print(f"      • {func['name']}({args_str}) (line {func['line']})")
            if len(result['functions']) > 3:
                print(f"      • ... and {len(result['functions']) - 3} more functions")
        
        # Key imports
        if result['imports']:
            key_imports = [imp for imp in result['imports'] if not imp['module'].startswith('.')][:3]
            if key_imports:
                print("   📦 Key imports:")
                for imp in key_imports:
                    if imp['type'] == 'import':
                        print(f"      • import {imp['module']}")
                    else:
                        print(f"      • from {imp['module']} import {imp['name']}")
        
        print()

if __name__ == '__main__':
    main()
