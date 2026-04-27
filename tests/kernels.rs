use anyhow::Result;
use cuda_async::device_operation::{DeviceOp, value};
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::half::f16;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::TileKernel;
use grout::kernels::add_2d_f16;
use std::sync::Arc;

#[test]
fn add_2d_kernel_compiles_and_executes() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            eprintln!("skipping CUDA kernel integration test: no CUDA devices found");
            return Ok(());
        }
        Err(err) => {
            eprintln!("skipping CUDA kernel integration test: CUDA unavailable: {err:?}");
            return Ok(());
        }
    }

    const BLOCK: usize = 4;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let lhs_host = Arc::new(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ]);
    let rhs_host = Arc::new(vec![
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
        f16::from_f32(40.0),
    ]);

    let lhs = Arc::new(
        api::copy_host_vec_to_device(&lhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let rhs = Arc::new(
        api::copy_host_vec_to_device(&rhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let out = api::zeros::<f16>(&[1, BLOCK]).sync_on(&stream)?;

    let result = add_2d_f16(value(out.partition([1, BLOCK])), value(lhs), value(rhs))
        .generics(vec![BLOCK.to_string()])
        .sync_on(&stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(&stream)?;

    let actual: Vec<f32> = actual.into_iter().map(|x| x.to_f32()).collect();
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}
