<script lang="ts">
	import SiteNav from '$lib/components/SiteNav.svelte';
	import { benchmarkReadmeUrl, docsUrl, protocolUrl, repoUrl, resultCardUrl } from '$lib/site';

	const heroFacts = [
		'Rust service',
		'REST + MCP',
		'PostgreSQL + pgvector',
		'Structured extraction'
	] as const;

	const heroImageUrl = '/hero-elephant.png';
</script>

<svelte:head>
	<meta
		property="og:title"
		content="Elephant | Long-term memory infrastructure for AI agents"
	/>
	<meta
		property="og:description"
		content="Structured extraction, entity resolution, multi-channel retrieval, and grounded reflection for multi-session AI agents."
	/>
</svelte:head>

<div id="top">
	<SiteNav />
</div>

<section class="hero-shell" aria-labelledby="hero-title">
	<div class="hero-topline">
		<p class="hero-kicker">Elephant / long-term memory infrastructure</p>
		<div class="hero-proof-inline">
			<span>LoCoMo benchmark</span>
			<strong>91.2%</strong>
		</div>
	</div>

	<div class="hero-stage">
		<div class="hero-copy">
			<h1 id="hero-title">
				<span class="hero-line-primary">Persistent memory</span>
				<span class="hero-line-secondary">for AI agents</span>
			</h1>
			<p class="hero-body">
				Elephant turns conversations into structured memory, resolves entities across sessions,
				and retrieves grounded context when agents need to answer.
			</p>
		</div>

		<div class="hero-art" aria-hidden="true">
			<img src={heroImageUrl} alt="" />
		</div>
	</div>

	<div class="hero-actions">
		<a class="button button-primary" href={repoUrl} target="_blank" rel="noreferrer"
			>View the repo</a
		>
		<a class="button button-secondary" href={docsUrl} target="_blank" rel="noreferrer"
			>Read the docs</a
		>
	</div>

	<div class="hero-divider" aria-hidden="true"></div>

	<div class="hero-foot">
		<ul class="hero-facts" aria-label="Elephant facts">
			{#each heroFacts as fact}
				<li>{fact}</li>
			{/each}
		</ul>

		<div class="hero-links">
			<a href={benchmarkReadmeUrl} target="_blank" rel="noreferrer">Benchmark README</a>
			<a href={protocolUrl} target="_blank" rel="noreferrer">Protocol</a>
			<a href={resultCardUrl} target="_blank" rel="noreferrer">Result card</a>
		</div>
	</div>
</section>
