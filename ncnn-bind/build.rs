extern crate bindgen;
use cmake::Config;

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::str;

fn output_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

fn ncnn_src_dir() -> PathBuf {
    // I'm using master
    let master_dir = "ncnn-master";
    // let tag_dir = format!("ncnn-src-{}", ncnn_tag());
    output_dir().join(master_dir)
}

fn _ncnn_tag() -> String {
    const DEFAULT_NCNN_TAG: &'static str = "20220729";
    env::var("NCNN_TAG").unwrap_or(DEFAULT_NCNN_TAG.to_string())
}

fn fetch() -> io::Result<()> {
    let target_dir = ncnn_src_dir();

    if target_dir.exists() {
        return Ok(());
    }

    let status = Command::new("git")
        .arg("clone")
        .arg("--recursive")
        .arg("--depth=1")
        .arg("-b")
        // .arg(ncnn_tag())
        .arg("master")
        .arg("https://github.com/Tencent/ncnn")
        .arg(&target_dir)
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "fetch failed"))
    }
}

fn build() -> io::Result<()> {
    let mut config = Config::new(ncnn_src_dir());
    config.define("NCNN_BUILD_TOOLS", "OFF");
    config.define("NCNN_BUILD_EXAMPLES", "OFF");
    config.define("NCNN_BUILD_BENCHMARK", "OFF");
    config.define("CMAKE_BUILD_TYPE", "Release");

    if cfg!(feature = "vulkan") {
        config.define("NCNN_VULKAN", "ON");
    }

    if use_dynamic_linking() {
        config.define("NCNN_SHARED_LIB", "ON");
    }

    let dst = config.build();

    println!("cargo:rustc-link-search=native={}", dst.display());

    Ok(())
}

fn search_include(include_paths: &[PathBuf], header: &str) -> String {
    for dir in include_paths {
        let include = dir.join(header);
        if fs::metadata(&include).is_ok() {
            return include.as_path().to_str().unwrap().to_string();
        }
    }
    format!("/usr/include/{}", header)
}

fn use_dynamic_linking() -> bool {
    if cfg!(feature = "static") && cfg!(feature = "dynamic") {
        panic!(
            "Both `static` and `dynamic` features are specified. Only one can be used at a time."
        );
    } else if cfg!(feature = "static") {
        false
    } else if cfg!(feature = "dynamic") {
        true
    } else {
        // By default use static linking for windows and dynamic for linux
        if cfg!(windows) {
            false
        } else {
            true
        }
    }
}

fn link_vulkan() {
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    let target_pointer_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap();

    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Ok(var) = env::var("VULKAN_SDK") {
        let suffix = match (&*target_family, &*target_pointer_width) {
            ("windows", "32") => "Lib32",
            ("windows", "64") => "Lib",
            _ => "lib",
        };
        println!("cargo:rustc-link-search={}/{}", var, suffix);
    }
    let lib = match &*target_family {
        "windows" => "vulkan-1",
        _ => "vulkan",
    };
    println!("cargo:rustc-link-lib={}", lib);
}

fn main() {
    let include_paths: Vec<PathBuf>;
    if cfg!(feature = "build") {
        println!("cargo:rerun-if-env-changed=NCNN_DIR");
        println!("cargo:rerun-if-env-changed=NCNN_TAG");

        include_paths = if let Ok(ncnn_dir) = env::var("NCNN_DIR") {
            // use prebuild ncnn dir
            let dir = PathBuf::from(ncnn_dir);
            println!(
                "cargo:rustc-link-search=native={}",
                dir.join("lib").to_string_lossy()
            );
            vec![dir.join("include").join("ncnn")]
        } else {
            // fetch from github and build
            fetch().unwrap();
            build().unwrap();

            println!(
                "cargo:rustc-link-search=native={}",
                output_dir().join("lib").to_string_lossy()
            );

            vec![output_dir().join("include").join("ncnn")]
        };
    } else {
        // how could Windows users find ncnn?
        // I don't care!
        include_paths = vec![
            PathBuf::from("/usr/include/ncnn"),
            PathBuf::from("/usr/local/include/ncnn"),
        ];
        println!("cargo:rustc-link-search=native={}", "/usr/local/lib");
        println!("cargo:rustc-link-search=native={}", "/usr/lib");
    }

    // have to link stdc++ explicitly
    // and HAVE to be after ncnn
    // `openmp-sys` could be helpful as well
    if use_dynamic_linking() {
        println!("cargo:rustc-link-lib=dylib=ncnn");
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        println!("cargo:rustc-link-lib=static=ncnn");
        println!("cargo:rustc-link-lib=static=gomp");
        println!("cargo:rustc-link-lib=static=stdc++");
    }

    if !cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

    if cfg!(feature = "vulkan") {
        link_vulkan();
    }

    let header = search_include(&include_paths, "c_api.h");

    let bindings = bindgen::Builder::default()
        .header(header)
        .allowlist_type("regex")
        .allowlist_function("ncnn.*")
        .allowlist_var("NCNN.*")
        .allowlist_type("ncnn.*")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(output_dir().join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
