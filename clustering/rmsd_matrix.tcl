set pdbid ""

set out2 [open "rmsd/rmsd_matrix_$pdbid.dat" w]

	for {set f 1} {$f < 201} {incr f} {
		set sys [mol new protein_topology/$pdbid/protein.prmtop]
		mol addfile plas7k/$pdbid/$pdbid.pw.frame_$f.mol2

		set out1 [open "protein_topology/$pdbid/bb_rmsd2_$f.dat" w]
		set A($f) ""
		set bb1 [atomselect $sys "(protein and (backbone and not hydrogen))"]

		for {set g $f} {$g < 201} {incr g} {
			
			set sys1 [mol new protein_topology/$pdbid/protein.prmtop]
			mol addfile plas7k/$pdbid/$pdbid.pw.frame_$g.mol2

			set bb2 [atomselect $sys1 "(protein and (backbone and not hydrogen))"]
 	                set mat1 [measure fit $bb2 $bb1]
        	        $bb2 move $mat1
                	set rmsdval1 [measure rmsd $bb2 $bb1]
                	puts $out1 "$rmsdval1"
			lappend A($f) $rmsdval1 

			unset rmsdval1 mat1 bb2
			mol delete $sys1
		}

		puts $out2 "$A($f)"
		close $out1
		unset bb1
		mol delete $sys
		exec rm "protein_topology/$pdbid/bb_rmsd2_$f.dat"

	}
	mol delete $sys

close $out2

exit