function run_becas(becas_src, folder, outfile)
% RUN_BECAS  Run a local, licensed BECAS on the 2D input in FOLDER and write its stiffness
% matrix and centres to OUTFILE (text, read by becas_io.read_becas_results).
%
%   run_becas('/path/to/BECAS/src/matlab', 'case_folder', 'becas_out.txt')
%
% This driver is ours; it only calls BECAS's documented functions (BECAS user guide,
% chapter "Using BECAS"). BECAS itself is never part of this repository.
addpath(genpath(becas_src));
options.foldername = folder;
utils = BECAS_Utils(options);
[Ks, ~] = BECAS_Constitutive_Ks(utils);
props = BECAS_CrossSectionProps(Ks, utils);
f = fopen(outfile, 'w');
fprintf(f, 'K=\n');
for i = 1:6
    fprintf(f, ' %.16e', Ks(i, :));
    fprintf(f, '\n');
end
fprintf(f, '\nShearX=%.16e\nShearY=%.16e\n', props.ShearX, props.ShearY);
fprintf(f, 'ElasticX=%.16e\nElasticY=%.16e\n', props.ElasticX, props.ElasticY);
fprintf(f, 'elements=%d\nnodes=%d\n', utils.ne_2d, utils.nn_2d);
fclose(f);
end
