echo "Starting calculation at $(date)"

files_dir="../plas20k"

# Iterate through directories in the "files/" directory
for dir_name in "$files_dir"/*; do\
    pdbid=$(basename "$dir_name")

    if [[ ! "$pdbid" =~ ^[6-7][0-9s-z] ]]; then
        continue
    fi

    echo "Processing PDB ID: $pdbid"

    sed -e "s|set pdbid \"\"|set pdbid \"$pdbid\"|" \
        rmsd_matrix.tcl > rmsd_matrix_tmp_6s.tcl

    /opt/vmd-1.9.3/bin/vmd -dispdev text -e rmsd_matrix_tmp_6s.tcl

    rm rmsd_matrix_tmp_6s.tcl
done

echo
echo "Matrix calculations complete."