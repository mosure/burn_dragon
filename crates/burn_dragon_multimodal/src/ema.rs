use crate::config::TargetTeacherConfig;
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub fn init_momentum_teacher<B, M>(student: &M, config: &TargetTeacherConfig) -> Option<M>
where
    B: Backend,
    M: Module<B> + Clone,
{
    if config.enabled {
        Some(student.clone().no_grad())
    } else {
        None
    }
}

pub fn ema_update_module<B, M>(teacher: M, student: &M, decay: f32) -> M
where
    B: Backend,
    M: Module<B> + Clone,
{
    struct EmaBlendMapper {
        decay: f32,
    }

    impl<B: Backend> ModuleMapper<B> for EmaBlendMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let teacher_tensor = param.val().detach();
            let decay = self.decay;
            param.load_mapper(move |student_tensor| {
                teacher_tensor.clone().mul_scalar(decay) + student_tensor.mul_scalar(1.0 - decay)
            })
        }
    }

    let mut mapper = EmaBlendMapper {
        decay: decay.clamp(0.0, 0.999_999),
    };
    let teacher = teacher.map(&mut mapper);
    let student_record = student.clone().no_grad().into_record();
    teacher.load_record(student_record).no_grad()
}

pub fn sync_optional_teacher_from_student<B, M>(
    teacher: Option<M>,
    student: &M,
    config: &TargetTeacherConfig,
) -> Option<M>
where
    B: Backend,
    M: Module<B> + Clone,
{
    if !config.enabled {
        return None;
    }
    Some(match teacher {
        Some(teacher) => ema_update_module::<B, _>(teacher, student, config.decay),
        None => student.clone().no_grad(),
    })
}
