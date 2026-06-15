import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import pytest

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.visnet.layer import VisnetLayer
from mlip.models.visnet.self_interaction import VisnetSelfInteractionBlock


class TestVisnetSelfInteractionBlock:
    l_max = 2
    irrep_dim = ((l_max + 1) ** 2) - 1
    vec_channels = 6
    num_species = 4
    n_nodes = 10

    key = jax.random.PRNGKey(0)

    def module(self, correlation: int) -> VisnetSelfInteractionBlock:
        return VisnetSelfInteractionBlock(
            vec_channels=self.vec_channels,
            l_max=self.l_max,
            correlation=correlation,
            num_species=self.num_species,
        )

    def inputs(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        scalar_feats = jax.random.normal(
            jax.random.fold_in(self.key, 0), (self.n_nodes, self.vec_channels)
        )
        vector_feats = jax.random.normal(
            jax.random.fold_in(self.key, 1),
            (self.n_nodes, self.irrep_dim, self.vec_channels),
        )
        species = jax.random.randint(
            jax.random.fold_in(self.key, 2), (self.n_nodes,), 0, self.num_species
        )
        return scalar_feats, vector_feats, species

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_output_shapes(self, correlation: int):
        scalar_feats, vector_feats, species = self.inputs()
        module = self.module(correlation)
        params = module.init(self.key, scalar_feats, vector_feats, species)
        scalars, vectors = jax.jit(module.apply)(
            params, scalar_feats, vector_feats, species
        )

        assert scalars.shape == (self.n_nodes, self.vec_channels)
        assert vectors.shape == (self.n_nodes, self.irrep_dim, self.vec_channels)

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_rot_equivariance(self, correlation: int):
        scalar_feats, vector_feats, species = self.inputs()
        module = self.module(correlation)
        params = module.init(self.key, scalar_feats, vector_feats, species)
        apply = jax.jit(module.apply)

        rotation_matrix = e3nn.rand_matrix(self.key)
        irreps = e3nn.Irreps.spherical_harmonics(self.l_max)[1:]
        rotation_matrix_irreps = irreps.D_from_matrix(rotation_matrix)

        vmap_rot_matrix_over_channels = jax.vmap(
            lambda feats: feats @ rotation_matrix_irreps,
            in_axes=2,
            out_axes=2,
        )

        scalars, vectors = apply(params, scalar_feats, vector_feats, species)

        vector_feats_rot = vmap_rot_matrix_over_channels(vector_feats)
        scalars_rot, vectors_rot = apply(
            params, scalar_feats, vector_feats_rot, species
        )

        # l=0 outputs must be rotation-invariant.
        assert jnp.allclose(scalars, scalars_rot, atol=1e-4, rtol=1e-4)

        # l>=1 outputs must be equivariant.
        vectors_rot_expected = vmap_rot_matrix_over_channels(vectors)
        assert jnp.allclose(vectors_rot, vectors_rot_expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_finite_gradients(self, correlation: int):
        scalar_feats, vector_feats, species = self.inputs()
        module = self.module(correlation)
        params = module.init(self.key, scalar_feats, vector_feats, species)

        def loss_fn(params):
            scalars, vectors = module.apply(
                params, scalar_feats, vector_feats, species
            )
            return jnp.sum(scalars**2) + jnp.sum(vectors**2)

        grads = jax.grad(loss_fn)(params)
        for leaf in jax.tree.leaves(grads):
            assert not jnp.any(jnp.isnan(leaf))
            assert not jnp.any(jnp.isinf(leaf))


class TestVisnetLayerSelfInteraction:
    l_max = 2
    irrep_dim = ((l_max + 1) ** 2) - 1
    num_heads = 2
    num_channels = 6
    vec_channels = 6
    num_species = 4
    n_nodes = 10
    n_edges = 68
    activation = "relu"
    attn_activation = "sigmoid"
    graph_cutoff_angstrom = 5.0
    vecnorm_type = "none"

    key = jax.random.PRNGKey(0)

    def module(self, correlation: int, last_layer: bool = False) -> VisnetLayer:
        return VisnetLayer(
            num_heads=self.num_heads,
            num_channels=self.num_channels,
            vec_channels=self.vec_channels,
            activation=self.activation,
            attn_activation=self.attn_activation,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            vecnorm_type=self.vecnorm_type,
            last_layer=last_layer,
            l_max=self.l_max,
            correlation=correlation,
            num_species=self.num_species,
        )

    def input_graph(self) -> Graph:
        node_feats = jnp.ones((self.n_nodes, self.num_channels))
        edge_feats = jnp.ones((self.n_edges, self.num_channels))
        distances = jnp.ones((self.n_edges,))
        spherical_feats = jax.random.normal(
            jax.random.fold_in(self.key, 1), (self.n_edges, self.irrep_dim)
        )
        vector_feats = jax.random.normal(
            jax.random.fold_in(self.key, 2),
            (self.n_nodes, self.irrep_dim, self.vec_channels),
        )
        species = jax.random.randint(
            jax.random.fold_in(self.key, 5), (self.n_nodes,), 0, self.num_species
        )
        senders = jax.random.randint(
            jax.random.fold_in(self.key, 3), (self.n_edges,), 0, self.n_nodes
        )
        receivers = jax.random.randint(
            jax.random.fold_in(self.key, 4), (self.n_edges,), 0, self.n_nodes
        )
        graph = Graph(
            nodes=GraphNodes(positions=None, features={"species": species}),
            edges=GraphEdges(features={}),
            globals=GraphGlobals(cell=None, weight=None),
            senders=senders,
            receivers=receivers,
            n_node=None,
            n_edge=None,
        )
        graph = graph.update_node_features(
            latent_scalars=node_feats, latent_vectors=vector_feats
        )
        graph = graph.update_edge_features(
            latent=edge_feats,
            distances=distances,
            spherical_embedding=spherical_feats,
        )
        return graph

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    @pytest.mark.parametrize("last_layer", [False, True])
    def test_output_shapes(self, correlation: int, last_layer: bool):
        graph_in = self.input_graph()
        module = self.module(correlation, last_layer)
        params = module.init(self.key, graph_in)
        out_graph = jax.jit(module.apply)(params, graph_in)

        assert out_graph.nodes.features["latent_scalars"].shape == (
            self.n_nodes,
            self.num_channels,
        )
        assert out_graph.nodes.features["latent_vectors"].shape == (
            self.n_nodes,
            self.irrep_dim,
            self.vec_channels,
        )

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_rot_equivariance(self, correlation: int):
        graph_in = self.input_graph()
        module = self.module(correlation, last_layer=False)
        params = module.init(self.key, graph_in)
        apply = jax.jit(module.apply)

        rotation_matrix = e3nn.rand_matrix(self.key)
        irreps = e3nn.Irreps.spherical_harmonics(self.l_max)[1:]
        rotation_matrix_irreps = irreps.D_from_matrix(rotation_matrix)

        out_graph = apply(params, graph_in)

        spherical_feats = graph_in.edges.features["spherical_embedding"]
        spherical_feats_rot = spherical_feats @ rotation_matrix_irreps

        vmap_rot_matrix_over_channels = jax.vmap(
            lambda feats: feats @ rotation_matrix_irreps,
            in_axes=2,
            out_axes=2,
        )
        vector_feats_rot = vmap_rot_matrix_over_channels(
            graph_in.nodes.features["latent_vectors"]
        )

        graph_in_rot = graph_in.update_edge_features(
            spherical_embedding=spherical_feats_rot
        )
        graph_in_rot = graph_in_rot.update_node_features(
            latent_vectors=vector_feats_rot
        )

        out_graph_rot = apply(params, graph_in_rot)

        assert jnp.allclose(
            out_graph.nodes.features["latent_scalars"],
            out_graph_rot.nodes.features["latent_scalars"],
            atol=1e-4,
            rtol=1e-4,
        )

        out_vector_feats_rot = vmap_rot_matrix_over_channels(
            out_graph.nodes.features["latent_vectors"]
        )
        assert jnp.allclose(
            out_graph_rot.nodes.features["latent_vectors"],
            out_vector_feats_rot,
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_finite_gradients(self, correlation: int):
        graph_in = self.input_graph()
        module = self.module(correlation, last_layer=False)
        params = module.init(self.key, graph_in)

        def loss_fn(params):
            out_graph = module.apply(params, graph_in)
            return jnp.sum(out_graph.nodes.features["latent_scalars"] ** 2) + jnp.sum(
                out_graph.nodes.features["latent_vectors"] ** 2
            )

        grads = jax.grad(loss_fn)(params)
        for leaf in jax.tree.leaves(grads):
            assert not jnp.any(jnp.isnan(leaf))
            assert not jnp.any(jnp.isinf(leaf))
