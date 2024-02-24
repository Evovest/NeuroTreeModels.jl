push!(LOAD_PATH, "../src/")
using Documenter
using DocumenterVitepress
using NeuroTreeModels

pages = [
    "index" => "index.md",
    "Quick start" => "quick-start.md",
    "Design" => "design.md",
    "Models" => "models.md",
    "API" => "API.md",
    # "Tutorials" => [
    #     "Regression - Boston" => "tutorials/regression-boston.md",
    #     "Logistic - Titanic" => "tutorials/logistic-titanic.md",
    #     "Classification - IRIS" => "tutorials/classification-iris.md",
    # ]
]

# makedocs(
#     sitename="NeuroTreeModels",
#     authors="Jeremie Desgagne-Bouchard and contributors.",
#     format=Documenter.HTML(
#         sidebar_sitename=false,
#         edit_link="main",
#         assets=["assets/style.css"]
#     ),
#     modules=[NeuroTreeModels],
#     pages=pages,
#     warnonly=true,
#     draft=false,
#     source="src",
#     build=joinpath(@__DIR__, "build")
# )

# deploydocs(
#     repo="github.com/Evovest/NeuroTreeModels.jl.git",
#     target="build",
#     devbranch="main",
#     devurl="dev",
#     versions=["stable" => "v^", "v#.#", "dev" => "dev"],
# )

makedocs(
    sitename="NeuroTreeModels.jl",
    authors="Evovest and contributors.",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/Evovest/NeuroTreeModels.jl", # this must be the full URL!
        devbranch="main",
        devurl="dev";
    ),
    modules=[NeuroTreeModels],
    pages=pages,
    warnonly=true,
    draft=false,
    source="src",
    build=joinpath(@__DIR__, "build")
)

deploydocs(;
    repo="github.com/Evovest/NeuroTreeModels.jl", # this must be the full URL!
    target="build", # this is where Vitepress stores its output
    branch="gh-pages",
    devbranch="main",
    push_preview=true
)

# We manually obtain the Documenter deploy configuration,
# so we can use it to set Vitepress's settings.
# TODO: make this better / encapsulate it in `makedocs`
# so the user does not need to know!

# deploy_config = Documenter.auto_detect_deploy_system()
# deploy_decision = Documenter.deploy_folder(
#     deploy_config;
#     repo="github.com/Evovest/NeuroTreeModels.jl", # this must be the full URL!
#     devbranch="main",
#     devurl="dev",
#     push_preview=true,
# )

# VitePress relies on its config file in order to understand where files will exist.
# We need to modify this file to reflect the correct base URL, however, Documenter
# only knows about the base URL at the time of deployment.

# So, after building the Markdown, we need to modify the config file to reflect the
# correct base URL, and then build the VitePress site.
# folder = deploy_decision.subfolder
# println("Deploying to $folder")
# vitepress_config_file = joinpath(@__DIR__, "build", ".vitepress", "config.mts")
# config = read(vitepress_config_file, String)
# new_config = replace(
#     config,
#     "base: 'REPLACE_ME_WITH_DOCUMENTER_VITEPRESS_BASE_URL_WITH_TRAILING_SLASH'" => "base: '/NeuroTreeModels.jl/$(folder)$(isempty(folder) ? "" : "/")'"
# )
# write(vitepress_config_file, new_config)

# # Build the docs using `npm` - we are assuming it's installed here!
# haskey(ENV, "CI") && begin
#     cd(@__DIR__) do
#         run(`npm run docs:build`)
#     end
#     touch(joinpath(@__DIR__, "build", ".vitepress", "dist", ".nojekyll"))
# end

# deploydocs(;
#     repo="github.com/Evovest/NeuroTreeModels.jl", # this must be the full URL!
#     target="build", # this is where Vitepress stores its output
#     branch="gh-pages",
#     devbranch="main",
#     push_preview=true
# )
