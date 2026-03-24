<script lang="ts">
	import { docsUrl, repoUrl } from '$lib/site';
</script>

<nav class="site-nav" aria-label="Primary">
	<a class="nav-brand" href="#top">
		<span class="nav-brand-mark">E</span>
		<span>elephant</span>
	</a>

	<div class="nav-actions">
		<a class="button button-tertiary" href={docsUrl} target="_blank" rel="noreferrer">Docs</a>
		<a class="button button-secondary" href={repoUrl} target="_blank" rel="noreferrer"
			>GitHub</a
		>
	</div>
</nav>
