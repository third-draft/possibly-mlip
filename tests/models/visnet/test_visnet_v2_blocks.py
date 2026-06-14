import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import pytest

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.visnet.blocks import (
    VisnetEmbeddingBlock,
    VisnetMultiHeadReadoutBlock,
)


class TestVisnetBlocks:
    n_nodes = 10
    n_edges = 68
    n_channels = 8
    n_rbf = 4
    l_max = 2
    vecnorm_type = "none"
    trainable_rbf = False
    radial_basis = "expnorm"
    graph_cutoff_angstrom = 5.0
    num_species = 10
    num_charges = None
    key = jax.random.PRNGKey(0)
    activation = "silu"

    def create_graph_from_input(
        self,
        senders: jax.Array,
        receivers: jax.Array,
        edge_features_dict: dict[str, jax.Array] = {},
        node_features_dict: dict[str, jax.Array] = {},
    ) -> Graph:
        return Graph(
            nodes=GraphNodes(positions=None, features=node_features_dict),
            edges=GraphEdges(features=edge_features_dict),
            globals=GraphGlobals(cell=None, weight=None),
            senders=senders,
            receivers=receivers,
            n_node=None,
            n_edge=None,
        )

    def common_input(self):
        graph_definition_kwargs = {}
        graph_definition_kwargs.update(
            minval=0, maxval=self.n_nodes, shape=(self.n_edges,), key=self.key
        )
        senders = jax.random.randint(**graph_definition_kwargs)
        receivers = jax.random.randint(**graph_definition_kwargs)
        return senders, receivers

    def embedding_block_input(self):
        node_features_dict = {}
        edge_features_dict = {}
        senders, receivers = self.common_input()
        node_features_dict.update(species=jnp.zeros(self.n_nodes).astype(jnp.int32))
        edge_features_dict.update(vectors=jnp.ones((self.n_edges, 3)))
        return self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )

    def readout_block_input(self, vec_channels: int):
        node_features_dict = {}
        edge_features_dict = {}
        senders, receivers = self.common_input()
        node_features_dict.update(
            latent_scalars=jnp.ones((self.n_nodes, self.n_channels)),
            latent_vectors=jnp.ones((
                self.n_nodes,
                (self.l_max + 1) ** 2 - 1,
                vec_channels,
            )),
        )
        edge_features_dict.update(
            latent=jnp.ones((self.n_edges, self.n_channels)),
        )
        return self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )

    @pytest.mark.parametrize("vec_channels", [None, 4])
    def test_visnet_embedding_block(self, vec_channels):
        graph_in = self.embedding_block_input()
        block = VisnetEmbeddingBlock(
            l_max=self.l_max,
            num_channels=self.n_channels,
            vec_channels=vec_channels,
            num_rbf=self.n_rbf,
            radial_basis=self.radial_basis,
            trainable_rbf=self.trainable_rbf,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            num_species=self.num_species,
            num_charges=self.num_charges,
            activation_fn=self.activation,
        )
        params = block.init(self.key, graph_in)
        graph = block.apply(params, graph_in)

        expected_vec_channels = (
            vec_channels if vec_channels is not None else self.n_channels
        )

        assert graph.nodes.features["embedding_scalars"].shape == (
            self.n_nodes,
            self.n_channels,
        )
        assert graph.edges.features["embedding"].shape == (
            self.n_edges,
            self.n_channels,
        )
        assert graph.nodes.features["embedding_vectors"].shape == (
            self.n_nodes,
            (self.l_max + 1) ** 2 - 1,
            expected_vec_channels,
        )

    def test_visnet_embedding_block_rot_equivariance(self):
        vec_channels = 4
        graph_in = self.embedding_block_input()
        vector_features = graph_in.edges.features["vectors"]
        block = VisnetEmbeddingBlock(
            l_max=self.l_max,
            num_channels=self.n_channels,
            vec_channels=vec_channels,
            num_rbf=self.n_rbf,
            radial_basis=self.radial_basis,
            trainable_rbf=self.trainable_rbf,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            num_species=self.num_species,
            num_charges=self.num_charges,
            activation_fn=self.activation,
        )
        params = block.init(self.key, graph_in)
        apply = jax.jit(block.apply)
        graph_out = apply(params, graph_in)

        rotation_matrix = e3nn.rand_matrix(self.key)
        vector_features_rot = vector_features @ rotation_matrix
        graph_in_rot = graph_in.update_edge_features(vectors=vector_features_rot)
        graph_out_rot = apply(params, graph_in_rot)

        assert jnp.allclose(
            graph_out.edges.features["embedding"],
            graph_out_rot.edges.features["embedding"],
            atol=1e-6,
        )
        assert jnp.allclose(
            graph_out.nodes.features["embedding_scalars"],
            graph_out_rot.nodes.features["embedding_scalars"],
            atol=1e-6,
        )
        assert graph_out.nodes.features["embedding_vectors"].shape == (
            self.n_nodes,
            (self.l_max + 1) ** 2 - 1,
            vec_channels,
        )

        spherical_features = graph_out.edges.features["spherical_embedding"]
        irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        rotation_matrix_irreps = irreps[1:].D_from_matrix(rotation_matrix)
        spherical_features_irreps_rot = spherical_features @ rotation_matrix_irreps
        assert jnp.allclose(
            spherical_features_irreps_rot,
            graph_out_rot.edges.features["spherical_embedding"],
            atol=1e-6,
        )

    @pytest.mark.parametrize("vec_channels", [8, 4])
    def test_visnet_readout_block(self, vec_channels):
        graph_in = self.readout_block_input(vec_channels)
        block = VisnetMultiHeadReadoutBlock(
            num_heads=2,
            num_channels=self.n_channels,
            vec_channels=vec_channels,
            activation=self.activation,
            vecnorm_type=self.vecnorm_type,
            l_max=self.l_max,
            predict_partial_charges=False,
        )
        params = block.init(self.key, graph_in)
        graph = block.apply(params, graph_in)

        assert graph.nodes.features["outputs"].shape == (self.n_nodes, 2, 1)
