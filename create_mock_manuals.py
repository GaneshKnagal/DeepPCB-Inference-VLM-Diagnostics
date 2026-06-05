import os

def create_manuals():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(ROOT_DIR, "mock_manuals")
    os.makedirs(output_dir, exist_ok=True)
    
    manuals = {
        "manual_open_circuits.txt": """
TECHNICAL SERVICE MANUAL: DEFECT CODE 001 - OPEN CIRCUIT
==============================================================
SECTION 1.0: DEFINITION & SPECIFICATIONS
An "Open Circuit" (YOLO Class 0) refers to a complete break or discontinuity in a conductive trace on a printed circuit board (PCB) or wafer interconnect. 
Custom inspection parameters mandate trace verification of 100% path continuity.

SECTION 2.0: CORE ROOT CAUSES
Based on field engineering records, open circuits are primarily caused by:
1. Over-etching: The board or wafer remains in the acid/etching bath too long, completely stripping thin copper traces.
2. Photolithography Mask Dust: A dust particle on the exposure mask blocks UV light, preventing the photoresist from hardening at that point.
3. Physical Scratches: Robot handler arms scratching the copper layer before deposition or etching.

SECTION 3.0: TROUBLESHOOTING & CORRECTION PROCEDURES
If the inspection system flags high rates of Open Circuits:
- STEP 1: Verify the Etch-Bath Timing. If timing deviates by >1.5%, recalibrate the automated timer.
- STEP 2: Inspect cleanroom Class-100 filters in the photolithography bay. High particle counts directly correlate with mask-level dust defects.
- STEP 3: Clean the UV projection lenses with cleanroom optical swabs and isopropyl alcohol.
- STEP 4: Inspect Handler Spatula #3. If rubber pads are worn down to metal, replace immediately to prevent physical scratch opens.
""",
        
        "manual_short_circuits.txt": """
TECHNICAL SERVICE MANUAL: DEFECT CODE 002 - SHORT CIRCUIT
==============================================================
SECTION 1.0: DEFINITION & SPECIFICATIONS
A "Short Circuit" (YOLO Class 1) refers to an unintended electrical connection (bridge) between two adjacent conductive traces.
Custom inspection parameters mandate trace isolation checks.

SECTION 2.0: CORE ROOT CAUSES
Short circuits are highly correlated with process failures in the etching and deposition stages:
1. Under-etching: Insufficient chemical concentration or low etching bath temperatures leave copper residues that bridge adjacent traces.
2. Solder Mask Alignment Error: Photomask shifting during mask application exposes trace gaps, allowing solder or copper plating to bridge.
3. Sputtering Overspray: Metallic sputtering deposition bleeding due to loose physical mask clamp fittings.

SECTION 3.0: TROUBLESHOOTING & CORRECTION PROCEDURES
If the inspection system flags high rates of Short Circuits:
- STEP 1: Measure acid concentration in the primary etching tank. Replenish etchant chemicals if concentration drops below 95% threshold.
- STEP 2: Verify the etching bath heater core. Temperature must remain stable at exactly 48.5°C (+/- 0.5°C).
- STEP 3: Recalibrate the mask aligner visual feedback cameras. Perform a X-Y mechanical recalibration of the alignment stage.
- STEP 4: Tighten the physical substrate clamps in sputtering chamber B.
""",

        "manual_mousebites.txt": """
TECHNICAL SERVICE MANUAL: DEFECT CODE 003 - MOUSEBITE
===========================================================
SECTION 1.0: DEFINITION & SPECIFICATIONS
A "Mousebite" (YOLO Class 2) refers to a semi-circular chunk or bite missing from the edge of a conductive trace. While not causing a complete open, it restricts current flow, creating hot-spots and reliability failures.

SECTION 2.0: CORE ROOT CAUSES
1. Photoresist Air Bubbles: Micro-bubbles trapped in the liquid photoresist layer during spin-coating leave localized spots unshielded during UV exposure.
2. Nitrogen Nozzle Pressure: High nitrogen blow pressure during cleanroom drying stages tearing off partially developed photoresist edges.
3. Mechanical Edge Abrasions: Robotic handlers gripping substrates too tightly, creating microscopic localized stress fractures.

SECTION 3.0: TROUBLESHOOTING & CORRECTION PROCEDURES
If the inspection system flags high rates of Mousebites:
- STEP 1: Inspect the photoresist spin-coater dispense nozzle. Clean nozzle tip to prevent cavitation and bubble creation during dispensing.
- STEP 2: Decrease nitrogen drying nozzle pressure from 3.2 bar to 2.8 bar.
- STEP 3: Perform load-cell calibration on Handler Grids A and B. Grip force must not exceed 1.2 Newtons.
""",

        "manual_spurs.txt": """
TECHNICAL SERVICE MANUAL: DEFECT CODE 004 - SPUR
=====================================================
SECTION 1.0: DEFINITION & SPECIFICATIONS
A "Spur" (YOLO Class 3) is a localized trace protrusion extending from a conductive line towards an adjacent line, significantly narrowing the spacing gap.

SECTION 2.0: CORE ROOT CAUSES
Spur formation is usually chemical or optical:
1. Localized Under-Etching: Inadequate chemical agitation causes stagnant chemical layers, leaving copper un-etched at the base of traces.
2. Photoresist Adhesion failure: Incomplete peeling of photoresist margins due to suboptimal soft-bake temperatures.

SECTION 3.0: TROUBLESHOOTING & CORRECTION PROCEDURES
If the inspection system flags high rates of Spurs:
- STEP 1: Increase the etchant spray nozzle pressure by 10% to improve local fluid dynamics and chemical agitation.
- STEP 2: Verify soft-bake oven temperature profiling. Ensure pre-exposure soft-bake reaches exactly 95°C for 60 seconds.
- STEP 3: Increase stripping chemical wash duration by 15 seconds.
""",

        "manual_spurious_copper.txt": """
TECHNICAL SERVICE MANUAL: DEFECT CODE 005 - SPURIOUS COPPER
================================================================
SECTION 1.0: DEFINITION & SPECIFICATIONS
"Spurious Copper" (YOLO Class 4) refers to isolated, non-functional islands of copper remaining on the non-conductive substrate material, completely detached from any valid traces.

SECTION 2.0: CORE ROOT CAUSES
1. Rinsing Contamination: Saturated rinse water containing dissolved copper re-depositing metallic copper particles during the post-etch wash.
2. Substrate Lamination Defects: Micro-cracks in the raw substrate fiberglass/dielectric capturing copper molecules that resist standard etching.

SECTION 3.0: TROUBLESHOOTING & CORRECTION PROCEDURES
If the inspection system flags high rates of Spurious Copper:
- STEP 1: Test water resistivity in the post-etch rinse tank. If resistivity drops below 18 Megohm-cm, replace the deionized (DI) water filters immediately.
- STEP 2: Check DI water circulation flow rates. Increase flow from 5L/min to 8L/min.
- STEP 3: Audit raw laminate substrate suppliers for surface roughness and micro-crack defects under SEM.
""",

        "manual_pinholes.txt": """
TECHNICAL SERVICE MANUAL: DEFECT CODE 006 - PIN-HOLE
=========================================================
SECTION 1.0: DEFINITION & SPECIFICATIONS
A "Pin-hole" (YOLO Class 5) is a microscopic void or missing circle inside a conductive trace or dielectric isolation layer.

SECTION 2.0: CORE ROOT CAUSES
1. Chemical Vapor Deposition (CVD) Cavities: Outgassing of volatile compounds from the substrate during high-vacuum CVD processes, leaving micro-voids.
2. Micro-bubbles in Plating baths: Hydrogen bubbles clinging to the substrate surface during electroplating, blocking copper deposition.

SECTION 3.0: TROUBLESHOOTING & CORRECTION PROCEDURES
If the inspection system flags high rates of Pin-holes:
- STEP 1: Check the vacuum pre-bake duration. Increase vacuum chamber pre-bake at 120°C from 10 minutes to 15 minutes to ensure full outgassing.
- STEP 2: Inspect electroplating bath surfactant concentration. Add anti-foaming agent or surfactant to reduce surface tension, allowing hydrogen bubbles to detach.
- STEP 3: Increase cathode agitation vibration rates in the electroplating bath.
"""
    }
    
    for filename, content in manuals.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content.strip())
            
    print(f"Generated {len(manuals)} technical troubleshooting manuals in: {output_dir}")

if __name__ == "__main__":
    create_manuals()
