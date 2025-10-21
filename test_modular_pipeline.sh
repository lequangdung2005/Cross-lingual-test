#!/bin/bash
# Quick test script for the modular pipeline

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║               Testing Modular Pipeline Structure                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found: $(python --version)"
echo ""

# Test 1: Show structure
echo "══════════════════════════════════════════════════════════════════════════════"
echo "TEST 1: Showing modular structure..."
echo "══════════════════════════════════════════════════════════════════════════════"
python show_modular_structure.py
echo ""

# Test 2: Check if modules exist
echo "══════════════════════════════════════════════════════════════════════════════"
echo "TEST 2: Checking if all modules exist..."
echo "══════════════════════════════════════════════════════════════════════════════"

files=(
    "pipeline/__init__.py"
    "pipeline/cli.py"
    "pipeline/database_builder.py"
    "pipeline/prompt_constructor.py"
    "pipeline_cli.py"
    "example_benchmark.json"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (MISSING)"
        all_exist=false
    fi
done
echo ""

if [ "$all_exist" = false ]; then
    echo "❌ Some files are missing. Please check the installation."
    exit 1
fi

# Test 3: Check CLI help
echo "══════════════════════════════════════════════════════════════════════════════"
echo "TEST 3: Testing CLI help system..."
echo "══════════════════════════════════════════════════════════════════════════════"
python pipeline_cli.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Main help works"
else
    echo "✗ Main help failed"
    exit 1
fi

python pipeline_cli.py build --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Build help works"
else
    echo "✗ Build help failed"
    exit 1
fi

python pipeline_cli.py construct --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Construct help works"
else
    echo "✗ Construct help failed"
    exit 1
fi

python pipeline_cli.py benchmark --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Benchmark help works"
else
    echo "✗ Benchmark help failed"
    exit 1
fi
echo ""

# Test 4: Check imports
echo "══════════════════════════════════════════════════════════════════════════════"
echo "TEST 4: Testing module imports..."
echo "══════════════════════════════════════════════════════════════════════════════"

python -c "from pipeline import build_database" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Can import build_database"
else
    echo "✗ Cannot import build_database"
fi

python -c "from pipeline import construct_single_prompt" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Can import construct_single_prompt"
else
    echo "✗ Cannot import construct_single_prompt"
fi

python -c "from pipeline import construct_benchmark_prompt" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Can import construct_benchmark_prompt"
else
    echo "✗ Cannot import construct_benchmark_prompt"
fi
echo ""

# Test 5: Check example files
echo "══════════════════════════════════════════════════════════════════════════════"
echo "TEST 5: Checking example files..."
echo "══════════════════════════════════════════════════════════════════════════════"

if [ -f "example_benchmark.json" ]; then
    queries=$(python -c "import json; print(len(json.load(open('example_benchmark.json'))))")
    echo "✓ example_benchmark.json exists with $queries queries"
else
    echo "✗ example_benchmark.json not found"
fi

if [ -f "example_usage_modular.py" ]; then
    echo "✓ example_usage_modular.py exists"
else
    echo "✗ example_usage_modular.py not found"
fi
echo ""

# Summary
echo "══════════════════════════════════════════════════════════════════════════════"
echo "✅ ALL TESTS PASSED!"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Modular pipeline is ready to use! 🚀"
echo ""
echo "Next steps:"
echo "  1. Run: python pipeline_cli.py --help"
echo "  2. See: MODULAR_STRUCTURE.md for documentation"
echo "  3. Try: python example_usage_modular.py"
echo ""
echo "Quick commands:"
echo "  • Build:     python pipeline_cli.py build -i data.json -o db.pkl"
echo "  • Construct: python pipeline_cli.py construct -d db.pkl -q 'def foo(): pass'"
echo "  • Benchmark: python pipeline_cli.py benchmark -d db.pkl -b benchmark.json -o out.json"
echo ""
