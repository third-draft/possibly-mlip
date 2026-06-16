import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.visnet.layer import VisnetLayer
from mlip.models_v1.visnet.models import VisnetLayer as VisnetLayerV1


class TestVisnetLayer:
    l_max = 2
    num_heads = 2
    num_channels = 6
    num_rbf = 4
    activation = "relu"
    attn_activation = "sigmoid"
    graph_cutoff_angstrom = 5.0
    vecnorm_type = "none"
    last_layer = False

    key = jax.random.PRNGKey(0)

    def module_v1(self) -> VisnetLayerV1:
        return VisnetLayerV1(
            num_heads=self.num_heads,
            num_channels=self.num_channels,
            activation=self.activation,
            attn_activation=self.attn_activation,
            cutoff=self.graph_cutoff_angstrom,
            vecnorm_type=self.vecnorm_type,
            last_layer=self.last_layer,
        )

    def module_v2(self) -> VisnetLayer:
        return VisnetLayer(
            num_heads=self.num_heads,
            num_channels=self.num_channels,
            activation=self.activation,
            attn_activation=self.attn_activation,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            vecnorm_type=self.vecnorm_type,
            last_layer=self.last_layer,
            l_max=self.l_max,
        )

    def input_dict(self) -> dict[str, jnp.ndarray]:
        n_nodes = 10
        n_edges = 68
        node_feats = jnp.ones(shape=(n_nodes, self.num_channels))
        edge_feats = jnp.ones(shape=(n_edges, self.num_channels))
        distances = jnp.ones(shape=(n_edges,))
        irrep_dim = ((self.l_max + 1) ** 2) - 1
        d_ij = jnp.ones(shape=(n_edges, irrep_dim))
        vector_feats = jnp.zeros(shape=(n_nodes, irrep_dim, self.num_channels))
        graph_definition_kwargs = {}
        graph_definition_kwargs.update(
            minval=0, maxval=n_nodes, shape=(n_edges,), key=self.key
        )
        senders = jax.random.randint(**graph_definition_kwargs)
        receivers = jax.random.randint(**graph_definition_kwargs)
        return {
            "latent_scalars": node_feats,
            "latent": edge_feats,
            "latent_vectors": vector_feats,
            "distances": distances,
            "senders": senders,
            "receivers": receivers,
            "d_ij": d_ij,
        }

    def input_v1(self) -> tuple[jnp.ndarray, ...]:
        in_dict = self.input_dict()
        inputs_v1 = tuple(value for value in in_dict.values())
        return inputs_v1

    def input_v2(self) -> tuple[Graph, ...]:
        full_input_dict = self.input_dict().copy()
        # convert d_ij to spherical_feats
        spherical_feats_dict = {"spherical_embedding": full_input_dict.pop("d_ij")}
        full_input_dict.update(spherical_feats_dict)
        graph = Graph(
            nodes=GraphNodes(positions=None, features={}),
            edges=GraphEdges(features={}),
            globals=GraphGlobals(
                cell=None,
                weight=None,
            ),
            senders=full_input_dict.pop("senders"),
            receivers=full_input_dict.pop("receivers"),
            n_node=None,
            n_edge=None,
        )
        node_inputs = {
            "latent_scalars": full_input_dict.pop("latent_scalars"),
            "latent_vectors": full_input_dict.pop("latent_vectors"),
        }
        graph = graph.update_node_features(**node_inputs)
        graph = graph.update_edge_features(**full_input_dict)
        return (graph,)

    def test_output_consistent_across_versions(self, standardize_params):
        params_v1 = standardize_params(
            self.module_v1().init(self.key, *self.input_v1())
        )
        params_v2 = standardize_params(
            self.module_v2().init(self.key, *self.input_v2())
        )
        result_v1 = jax.jit(self.module_v1().apply)(params_v1, *self.input_v1())
        result_v2 = jax.jit(self.module_v2().apply)(params_v2, *self.input_v2())

        # extract comparable results from both versions ( we want to compare
        # node_feats, edge_feats, vector_feats)

        # V1:
        node_feats_v1 = result_v1[0]
        edge_feats_v1 = result_v1[1]
        vector_feats_v1 = result_v1[2]
        out_node_feats_v1 = self.input_dict()["latent_scalars"] + node_feats_v1
        out_edge_feats_v1 = self.input_dict()["latent"] + edge_feats_v1
        out_vector_feats_v1 = self.input_dict()["latent_vectors"] + vector_feats_v1

        # V2:
        out_node_feats_v2 = result_v2.nodes.features["latent_scalars"]
        out_edge_feats_v2 = result_v2.edges.features["latent"]
        out_vector_feats_v2 = result_v2.nodes.features["latent_vectors"]

        # compare the results
        assert jnp.allclose(out_node_feats_v1, out_node_feats_v2, atol=1e-6)
        assert jnp.allclose(out_edge_feats_v1, out_edge_feats_v2, atol=1e-6)
        assert jnp.allclose(out_vector_feats_v1, out_vector_feats_v2, atol=1e-6)

    def test_visnet_layer_rot_equivariance(self):
        (in_graph,) = self.input_v2()
        module = self.module_v2()
        params = module.init(self.key, in_graph)
        rotation_matrix = e3nn.rand_matrix(self.key)
        apply = jax.jit(module.apply)

        out_graph = apply(params, in_graph)
        out_node_feats = out_graph.nodes.features["latent_scalars"]
        out_edge_feats = out_graph.edges.features["latent"]
        out_vector_feats = out_graph.nodes.features["latent_vectors"]

        # Out of all input features to the layer, only the spherical harmonics and
        # vector_feats not invariant to rotation. However vector_feats are set to zero
        # as input.
        # Rotate spherical_feats:
        spherical_feats = in_graph.edges.features["spherical_embedding"]
        irreps = e3nn.Irreps.spherical_harmonics(self.l_max)[1:]
        rotation_matrix_irreps = irreps.D_from_matrix(rotation_matrix)
        spherical_feats_rot = spherical_feats @ rotation_matrix_irreps
        graph_in_rot = in_graph.update_edge_features(
            spherical_embedding=spherical_feats_rot
        )

        out_graph_rot = apply(params, graph_in_rot)
        out_node_feats_rot = out_graph_rot.nodes.features["latent_scalars"]
        out_edge_feats_rot = out_graph_rot.edges.features["latent"]
        out_vector_feats_rot = out_graph_rot.nodes.features["latent_vectors"]

        # Compare rotation invariant outputs
        assert jnp.allclose(out_node_feats, out_node_feats_rot, atol=1e-6)
        assert jnp.allclose(out_edge_feats, out_edge_feats_rot, atol=1e-6)

        # Compare rotation equivariant outputs
        vmap_rot_matrix_over_channels = jax.vmap(
            lambda feats: feats @ rotation_matrix_irreps,
            in_axes=2,
            out_axes=2,
        )
        vector_feats_rot = vmap_rot_matrix_over_channels(out_vector_feats)
        assert jnp.allclose(out_vector_feats_rot, vector_feats_rot, atol=1e-6)


class TestVisnetLayerRelaxedEquivariance:
    """Tests for VisnetLayer with relaxed_equivariance=True."""

    l_max = 2
    num_heads = 2
    num_channels = 6
    num_rbf = 4
    activation = "relu"
    attn_activation = "sigmoid"
    graph_cutoff_angstrom = 5.0
    vecnorm_type = "none"
    key = jax.random.PRNGKey(42)

    def _make_graph(self):
        n_nodes, n_edges = 8, 30
        irrep_dim = (self.l_max + 1) ** 2 - 1
        senders = jax.random.randint(self.key, (n_edges,), 0, n_nodes)
        receivers = jax.random.randint(self.key, (n_edges,), 0, n_nodes)
        graph = Graph(
            nodes=GraphNodes(positions=None, features={}),
            edges=GraphEdges(features={}),
            globals=GraphGlobals(cell=None, weight=None),
            senders=senders,
            receivers=receivers,
            n_node=None,
            n_edge=None,
        )
        graph = graph.update_node_features(
            latent_scalars=jnp.ones((n_nodes, self.num_channels)),
            latent_vectors=jnp.zeros((n_nodes, irrep_dim, self.num_channels)),
        )
        graph = graph.update_edge_features(
            latent=jnp.ones((n_edges, self.num_channels)),
            distances=jnp.ones((n_edges,)),
            spherical_embedding=jnp.ones((n_edges, irrep_dim)),
        )
        return graph

    def _make_layer(self, relaxed_equivariance: bool) -> VisnetLayer:
        return VisnetLayer(
            num_heads=self.num_heads,
            num_channels=self.num_channels,
            activation=self.activation,
            attn_activation=self.attn_activation,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            vecnorm_type=self.vecnorm_type,
            last_layer=False,
            l_max=self.l_max,
            relaxed_equivariance=relaxed_equivariance,
        )

    def test_layer_with_relaxed_equivariance_produces_finite_output(self):
        graph = self._make_graph()
        graph = graph.update_global_features(relaxed_equiv_weight=0.05)
        layer = self._make_layer(relaxed_equivariance=True)
        params = layer.init(self.key, graph)
        out = jax.jit(layer.apply)(params, graph)
        assert jnp.all(jnp.isfinite(out.nodes.features["latent_scalars"]))
        assert jnp.all(jnp.isfinite(out.nodes.features["latent_vectors"]))

    def test_layer_with_beta_zero_matches_equivariant_baseline(self):
        """With beta=0 the relaxed block is a no-op: outputs match standard layer."""
        graph = self._make_graph()
        standard = self._make_layer(relaxed_equivariance=False)
        relaxed = self._make_layer(relaxed_equivariance=True)

        params_std = standard.init(self.key, graph)
        # Build matching params for the relaxed layer by sharing the
        # equivariant sub-tree and zero-initialising the relaxed block.
        params_rel = relaxed.init(self.key, graph)

        # Inject beta=0 — the block becomes an exact identity regardless of kernel.
        graph_beta0 = graph.update_global_features(relaxed_equiv_weight=0.0)

        out_std = jax.jit(standard.apply)(params_std, graph)
        out_rel = jax.jit(relaxed.apply)(params_rel, graph_beta0)

        # Both layers are freshly initialized from the same key so share
        # identical equivariant weights. With beta=0 outputs must agree.
        assert jnp.allclose(
            out_std.nodes.features["latent_scalars"],
            out_rel.nodes.features["latent_scalars"],
            atol=1e-5,
        )

    def test_nonzero_beta_changes_vector_output(self):
        """A non-zero beta with a non-zero kernel changes the vector output."""
        graph = self._make_graph()
        layer = self._make_layer(relaxed_equivariance=True)
        params_zero = layer.init(self.key, graph)
        # Randomise ALL parameters — the irrep_mixer kernel will be non-zero.
        params_rand = jax.tree.map(
            lambda p: jax.random.normal(self.key, p.shape) * 0.1, params_zero
        )

        out_beta0 = jax.jit(layer.apply)(
            params_rand, graph.update_global_features(relaxed_equiv_weight=0.0)
        )
        out_beta05 = jax.jit(layer.apply)(
            params_rand, graph.update_global_features(relaxed_equiv_weight=0.05)
        )
        assert not jnp.allclose(
            out_beta0.nodes.features["latent_vectors"],
            out_beta05.nodes.features["latent_vectors"],
            atol=1e-6,
        )
