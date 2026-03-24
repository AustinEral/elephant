<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Elephant | Long-term memory infrastructure for AI agents</title>
	<meta
		name="description"
		content="Elephant is long-term memory infrastructure for AI agents with structured extraction, retrieval, reflection, and benchmarked memory quality."
	/>
</svelte:head>

<div class="site-shell">
	<div class="page-frame">
		{@render children()}
	</div>
</div>
