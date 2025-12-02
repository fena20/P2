#!/bin/bash
# Script to list and display all generated figures and tables

echo "=========================================="
echo "EDGE AI RESEARCH - OUTPUTS SUMMARY"
echo "=========================================="
echo ""

echo "📊 FIGURES (7 total):"
echo "-------------------"
ls -lh visualization/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

echo "📋 TABLES (8 total):"
echo "-------------------"
ls -lh tables/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

echo "=========================================="
echo "TABLE CONTENTS PREVIEW"
echo "=========================================="
echo ""

echo "Table 1: Model Comparison"
echo "-------------------------"
head -5 tables/table1_model_comparison.csv 2>/dev/null | column -t -s','
echo ""

echo "Table 5: Edge Performance"
echo "-------------------------"
head -5 tables/table5_edge_performance.csv 2>/dev/null | column -t -s','
echo ""

echo "Table 6: Energy Savings"
echo "----------------------"
head -6 tables/table6_energy_savings.csv 2>/dev/null | column -t -s','
echo ""

echo "=========================================="
echo "FILE LOCATIONS"
echo "=========================================="
echo "Figures: $(pwd)/visualization/"
echo "Tables:  $(pwd)/tables/"
echo ""
echo "View complete table contents in: FIGURES_AND_TABLES.md"
echo "View HTML summary: view_outputs.html"
