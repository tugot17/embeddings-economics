#!/bin/bash

# SGLang Top Kernels Profiling Script
# Uses correct NCU syntax from official documentation

# Array of kernel patterns for filtering
declare -a kernels=(
    "nvjet_tst_192x176_64x4_1x2_h_bz_coopB_T"
    "nvjet_tst_256x144_64x4_2x1_v_bz_coopA_T" 
    "cutlass::device_kernel"
    "nvjet_tst_192x208_64x4_2x1_v_bz_coopB_T"
    "flashinfer::norm::RMSNormKernel"
    "flashinfer::activation::act_and_mu"
)

# Configuration
PYTHON_PATH="/home/ubuntu/embeddings-economics/profile_sglang/.venv/bin/python"
SCRIPT_PATH="embedding_profile.py"
NCU_PATH="/opt/nvidia/nsight-compute/2024.3.1/ncu"
PROFILE_SET="detailed"

# Create output directory
OUTPUT_DIR="kernel_profiles_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting kernel-specific profiling with correct NCU syntax..."
echo "Output directory: $OUTPUT_DIR"
echo "Profile set: $PROFILE_SET"
echo "=========================================="

# Function to clean kernel name for filename
clean_kernel_name() {
    echo "$1" | sed 's/[^a-zA-Z0-9_]/_/g' | sed 's/__*/_/g' | sed 's/_$//'
}

# Test basic NCU functionality first
echo "Testing basic NCU functionality..."
test_output="$OUTPUT_DIR/basic_test"

sudo "$NCU_PATH" \
    --launch-count 10 \
    --set basic \
    -o "$test_output" \
    "$PYTHON_PATH" "$SCRIPT_PATH"

if [ $? -eq 0 ]; then
    echo "✓ Basic NCU test successful! Proceeding with kernel filtering..."
    
    # Profile each kernel individually using --kernel-name with regex
    for i in "${!kernels[@]}"; do
        kernel="${kernels[$i]}"
        kernel_clean=$(clean_kernel_name "$kernel")
        output_file="$OUTPUT_DIR/kernel_${i}_${kernel_clean}"
        
        echo ""
        echo "Profiling kernel $((i+1))/6: $kernel"
        echo "Output: ${output_file}.ncu-rep"
        echo "----------------------------------------"
        
        # Use --kernel-name with the kernel pattern
        # According to docs, this supports regex matching
        sudo "$NCU_PATH" \
            --kernel-name "$kernel" \
            --set "$PROFILE_SET" \
            -o "$output_file" \
            "$PYTHON_PATH" "$SCRIPT_PATH"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully profiled: $kernel"
            
            if [ -f "${output_file}.ncu-rep" ]; then
                file_size=$(ls -lh "${output_file}.ncu-rep" | awk '{print $5}')
                echo "  Profile file size: $file_size"
            else
                echo "  Warning: No profile file generated (kernel may not have executed)"
            fi
        else
            echo "✗ Failed to profile: $kernel"
            echo "  Trying with regex prefix..."
            
            # Try with explicit regex: prefix if plain name fails
            sudo "$NCU_PATH" \
                --kernel-name "regex:$kernel" \
                --set "$PROFILE_SET" \
                -o "${output_file}_regex" \
                "$PYTHON_PATH" "$SCRIPT_PATH"
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Regex approach worked for: $kernel"
            else
                echo "  ✗ Both approaches failed for: $kernel"
            fi
        fi
        
        sleep 2
    done
    
    # Create combined profile using kernel-id with multiple patterns
    echo ""
    echo "Creating combined profile with all kernels using --kernel-id..."
    
    # Build combined kernel-id filter
    # Format: context-id:stream-id:[name-operator:]kernel-name:invocation-nr
    # We'll use ::regex: for regex matching across all contexts/streams
    all_kernels_regex=""
    for i in "${!kernels[@]}"; do
        escaped_kernel=$(echo "${kernels[$i]}" | sed 's/[[\.*^$()+?{|]/\\&/g')
        if [ $i -eq 0 ]; then
            all_kernels_regex="$escaped_kernel"
        else
            all_kernels_regex="$all_kernels_regex|$escaped_kernel"
        fi
    done
    
    combined_output="$OUTPUT_DIR/all_kernels_combined"
    echo "Combined regex pattern: $all_kernels_regex"
    
    sudo "$NCU_PATH" \
        --kernel-id "::regex:$all_kernels_regex" \
        --set "$PROFILE_SET" \
        -o "$combined_output" \
        "$PYTHON_PATH" "$SCRIPT_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ Combined profile created successfully"
    else
        echo "✗ Combined profile failed"
        echo "  Creating fallback profile with launch-count limit..."
        
        # Fallback: limited launch count to capture main kernels
        sudo "$NCU_PATH" \
            --launch-count 100 \
            --set "$PROFILE_SET" \
            -o "${combined_output}_fallback" \
            "$PYTHON_PATH" "$SCRIPT_PATH"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Fallback profile created"
        fi
    fi
    
else
    echo "✗ Basic NCU test failed! There may be a fundamental issue."
    echo "  Check if the Python script runs without NCU:"
    echo "  $PYTHON_PATH $SCRIPT_PATH"
    echo ""
    echo "  Check NCU version and help:"
    echo "  sudo $NCU_PATH --version"
    echo "  sudo $NCU_PATH --help"
    exit 1
fi

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR/"/*.ncu-rep 2>/dev/null || echo "No .ncu-rep files found"

echo ""
echo "To view results:"
echo "ncu-ui $OUTPUT_DIR/*.ncu-rep"
echo ""
echo "Target kernels to analyze:"
for kernel in "${kernels[@]}"; do
    echo "  - $kernel"
done

echo ""
echo "Usage tips:"
echo "- Open individual kernel profiles to see detailed metrics"
echo "- Use the combined profile to compare kernels side-by-side"  
echo "- Filter by kernel name in the NCU GUI for focused analysis"