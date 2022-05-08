use fyrox::{
    core::{algebra::Vector3, math::Vector3Ext, pool::Handle},
    engine::resource_manager::ResourceManager,
    scene::{node::Node, Scene, graph::Graph},
};

pub struct Weapon {
    model: Handle<Node>,
    shot_point: Handle<Node>,
    shot_timer: f32,
}

impl Weapon {
    pub async fn new(scene: &mut Scene, resource_manager: ResourceManager) -> Self {
        // Yeah, you need only few lines of code to load a model of any complexity.
        let model = resource_manager
            .request_model("data/models/m4/m4.FBX")
            .await
            .unwrap()
            .instantiate_geometry(scene);

        let shot_point = scene.graph.find_by_name(model, "Weapon:ShotPoint");

        Self {
            model,
            shot_point,
            shot_timer: 0.0,
        }
    }

    pub fn model(&self) -> Handle<Node> {
        self.model
    }

    pub fn shot_point(&self) -> Handle<Node> {
        self.shot_point
    }

    pub fn update(&mut self, dt: f32) {
        self.shot_timer = (self.shot_timer - dt).min(0.0);
    }

    pub fn can_shoot(&self) -> bool {
        self.shot_timer <= 0.0
    }

    pub fn shoot(&mut self) {
        self.shot_timer = 1.0;
    }
}