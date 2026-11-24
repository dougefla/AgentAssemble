# Local diagnostic helpers

Import with `from tools.image_compare import compare_views` (add `str(SKILL_DIR)` to `sys.path` if needed).

### `tools/image_compare.py::compare_views(reference_path, preview_path, colors, *, root, overlay_path=None)`

- **Parameters:** `reference_path`, `preview_path`: str or pathlib.Path of readable RGB images. `colors`: nonempty dict mapping string link names to finite numeric RGB triples in [0,1], taken from template/materials; colours must be chromatic, not grey. `root`: keyword-only string key in colors identifying the fixed root. `overlay_path`: keyword-only str/Path for a PNG output, or None (default) for no file. Use comparable camera azimuth/elevation for meaningful comparison. No world-frame quantities are accepted.
- **Returns:** Dict with `preview_to_reference_scale` (float), `preview_to_reference_offset_px` (two floats), `root_span_residual` (float relative root-span mismatch; large values indicate camera/occlusion mismatch), `links` (dict keyed by colors), `overlay_path` (str or None). Each visible link entry contains `reference_bounds_px`, `registered_preview_bounds_px` (2x2 lists [min,max] in reference pixel coordinates), `center_delta_px` (two floats, reference minus registered preview), `center_error_root_fraction` (float), `reference_over_preview_span` (two floats, target/preview X,Y spans), and `visible_in_reference`, `visible_in_preview` (bool). If absent in either view, only the two visibility fields are returned. Image +X is right, +Y down. Bounds use 1st/99th percentiles. Optional panel shows target, registered preview, then blend. These are projected visible regions, not 3D poses or automatic corrections.
- **Raises:** ValueError for invalid colours, achromatic palette, missing/degenerate root region; OSError for unreadable images or unwritable overlay. Does not reject wrong camera angles or occlusion; inspect residual/panel.
- **Example:** `compare_views(reference_path, preview_path, {'link_0': [0.84,0.26,0.26], 'link_1': [0.26,0.84,0.26]}, root='link_0', overlay_path=OUTPUT_DIR / 'comparison.png')` where reference_path/preview_path are existing image paths and OUTPUT_DIR is a pathlib.Path. Omitting overlay_path returns the same measurements without writing a panel.

### `tools/round_features.py::find_round_features(vertices, faces, *, min_segments=12)`

- **Parameters:** `vertices`: finite numeric N×3 list or numpy array, N>=3, in one mesh coordinate frame, without recentering. `faces`: M×3 list or numpy integer array of zero-based triangular vertex indices; an empty numpy integer array of shape (0,3) is allowed. `min_segments`: keyword-only integer >=8, default 12, minimum distinct rim vertex count. Exact duplicate vertices are welded internally. Coordinates may be normalized mesh units or metres; output uses the same frame/units. No mesh object, file path, faces with float dtype, or axis aliases are accepted.
- **Returns:** List of dicts, sorted by decreasing `radius`. Each has `center` (3 floats), `axis` (3 floats, unsigned unit circle-plane normal with largest-magnitude coordinate positive), `radius` (positive float), `relative_fit_error` (float RMS radial/planar error divided by radius, <=0.025), `vertex_count` (int), and `origin_line_distance` (float distance of fitted axis line from supplied coordinate origin). Centers/radii/distances use input units; axes use the input coordinate frame. Only isolated degree-two sharp-edge loops with full angular coverage are returned. Empty list means no supported circular rims, not no joint. Rounded/integrated hardware may be missed, and screw heads or decorative rims may be included: these are evidence, not an automatic joint solution.
- **Raises:** ValueError for nonfinite/wrong-shaped coordinates, invalid/noninteger triangle indices, or invalid min_segments. TypeError may occur for inputs not convertible to numeric arrays. Does not reject incorrect mechanical correspondence or choose axis sign.
- **Example:** `find_round_features([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]], [[0,1,4],[1,2,4],[2,3,4],[3,0,4]])` returns [] because the sharp-edge pyramid is not a sufficiently sampled circular rim. For mesh arrays use `find_round_features(mesh.vertices, mesh.faces, min_segments=12)`.

### `tools/preflight.py::run_preflight(*, skill_dir)`

- **Parameters:** `skill_dir`: keyword-only str or pathlib.Path of the skill directory. No simulator/model input; deterministic synthetic test.
- **Returns:** Dict `{'ok': True, 'checks': list[str]}` after exercising every public helper and documented calling form.
- **Raises:** AssertionError on failed synthetic geometry/image checks; import or I/O exceptions if dependencies or temporary storage are unavailable.
- **Example:** `run_preflight(skill_dir=SKILL_DIR)`
